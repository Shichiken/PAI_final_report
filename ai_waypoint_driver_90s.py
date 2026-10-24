#!/usr/bin/env python3
"""ROS 2 / Gazebo: decision-tree controlled 90-second L-shaped patrol.

This keeps the original AI starter's supervised decision-tree controller.
It repeats only the previously verified waypoints in a small open area.
The run stops after a configurable amount of real elapsed time (default 90 s).

Prerequisites: rclpy, nav_msgs, geometry_msgs, scikit-learn, Gazebo robot
`vehicle_blue` and active ROS <-> Gazebo odometry / command bridges.
It does not detect or avoid obstacles; do not run with people or obstacles.
"""
import argparse
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

CMD_TOPIC = '/model/vehicle_blue/cmd_vel'
ODOM_TOPIC = '/model/vehicle_blue/odometry'


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def make_policy():
    """Learn movement actions from simple synthetic expert labels."""
    features, labels = [], []
    for dist_step in range(0, 41):
        distance = dist_step * 0.1
        for degree in range(-180, 181, 5):
            error = math.radians(degree)
            if distance < 0.35:
                label = 'STOP'
            elif error > 0.18:
                label = 'LEFT'
            elif error < -0.18:
                label = 'RIGHT'
            else:
                label = 'FORWARD'
            features.append([distance, error])
            labels.append(label)

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels)
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(x_train, y_train)
    print('Decision tree test accuracy on synthetic examples: '
          f'{accuracy_score(y_test, model.predict(x_test)):.3f}', flush=True)
    return model


def l_shaped_waypoints(x, y, yaw):
    """Forward 1.5 m, left 1.0 m, backtrack, then return to start.

    All points are relative to the initial pose (not global fixed axes).
    Returning on the same segments avoids exploring new terrain.
    """
    forward = (math.cos(yaw), math.sin(yaw))
    left = (-math.sin(yaw), math.cos(yaw))
    a = (x + 1.5 * forward[0], y + 1.5 * forward[1])
    b = (a[0] + 1.0 * left[0], a[1] + 1.0 * left[1])
    return [a, b, a, (x, y)]


class AIWaypointDriver(Node):
    def __init__(self, duration_sec=90.0):
        super().__init__('ai_waypoint_driver_90s')
        self.duration_sec = duration_sec
        self.policy = make_policy()
        self.pub = self.create_publisher(Twist, CMD_TOPIC, 10)
        self.sub = self.create_subscription(Odometry, ODOM_TOPIC, self.on_odom, 10)
        self.timer = self.create_timer(0.1, self.control)

        self.pose = None
        self.goals = None
        self.index = 0
        self.reached_count = 0
        self.completed = False
        self.last_label = None
        self.start_time = None
        self.next_status_at = 5.0
        self.get_logger().info(
            f'Waiting for odometry: {ODOM_TOPIC}; target run {duration_sec:.0f} seconds')

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, yaw)
        if self.goals is None:
            self.goals = l_shaped_waypoints(p.x, p.y, yaw)
            self.start_time = time.monotonic()
            self.get_logger().info(
                f'Patrol route (relative to starting position): {self.goals}')
            self.get_logger().info(
                f'Patrol starts. Repeating these 4 goals for {self.duration_sec:.0f}s.')

    def publish_stop(self):
        self.pub.publish(Twist())

    def control(self):
        cmd = Twist()
        if self.completed or self.pose is None or self.goals is None:
            self.pub.publish(cmd)
            return

        elapsed = time.monotonic() - self.start_time
        if elapsed >= self.duration_sec:
            self.completed = True
            self.publish_stop()
            self.get_logger().info(
                f'Timed run completed: elapsed={elapsed:.1f}s; '
                f'waypoints_reached={self.reached_count}; '
                f'full_laps={self.reached_count // 4}. Robot stopped.')
            return

        x, y, yaw = self.pose
        gx, gy = self.goals[self.index]
        distance = math.hypot(gx - x, gy - y)
        desired_yaw = math.atan2(gy - y, gx - x)
        error = normalize_angle(desired_yaw - yaw)
        # AI model chooses the next action (synthetic rule imitation).
        label = self.policy.predict([[min(distance, 4.0), error]])[0]

        if label == 'STOP':
            self.reached_count += 1
            self.get_logger().info(
                f'Reached waypoint {self.index + 1}/4, '
                f'lap {((self.reached_count - 1) // 4) + 1}; '
                f'error={distance:.2f}m; elapsed={elapsed:.1f}s')
            self.index = (self.index + 1) % len(self.goals)
            # Send zero velocity briefly before turning to the next target.
        elif label == 'FORWARD':
            cmd.linear.x = min(0.30, max(0.10, 0.4 * distance))
        elif label == 'LEFT':
            cmd.angular.z = min(0.65, max(0.12, abs(error)))
        elif label == 'RIGHT':
            cmd.angular.z = -min(0.65, max(0.12, abs(error)))

        if label != self.last_label:
            self.get_logger().info(
                f'[elapsed {elapsed:5.1f}/{self.duration_sec:.0f}s] '
                f'Goal {self.index + 1}/4 action={label} '
                f'distance={distance:.2f}m angle_error={error:.2f}')
            self.last_label = label
        if elapsed >= self.next_status_at:
            self.get_logger().info(
                f'Progress: {elapsed:.1f}/{self.duration_sec:.0f}s, '
                f'waypoints_reached={self.reached_count}, action={label}')
            self.next_status_at += 5.0

        self.pub.publish(cmd)


def main():
    parser = argparse.ArgumentParser(description='AI guided Gazebo patrol (~90 seconds).')
    parser.add_argument('--duration', type=float, default=90.0,
                        help='Run duration in real seconds after first odometry (default: 90)')
    args = parser.parse_args()
    if not (5.0 <= args.duration <= 600.0):
        parser.error('duration must be in the range 5..600 seconds')

    rclpy.init()
    node = AIWaypointDriver(duration_sec=args.duration)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted. Sending stop command.')
    finally:
        node.completed = True  # Prevent the control timer from issuing any new motion.
        if rclpy.ok():
            for _ in range(3):
                node.publish_stop()
                rclpy.spin_once(node, timeout_sec=0.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
