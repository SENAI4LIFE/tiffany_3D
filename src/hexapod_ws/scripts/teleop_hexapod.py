#!/usr/bin/env python3
import sys
import termios
import tty
import select

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

POSE_STEP = 1.5
POSE_MAX  = 15.0

MOVE_KEYS = {'w', 's', 'a', 'd', '\x1b[A', '\x1b[B', '\x1b[C', '\x1b[D'}

class TeleopHexapod(Node):
    def __init__(self):
        super().__init__('teleop_hexapod')

        self.vel_pub   = self.create_publisher(Twist,  '/cmd_vel',       10)
        self.state_pub = self.create_publisher(String, '/tiffany/state', 10)

        self.linear_speed  = 0.15
        self.angular_speed = 1.0

        self.pose_roll  = 0.0
        self.pose_pitch = 0.0
        self.pose_mode  = False
        self.current_lx = 0.0
        self.current_ly = 0.0
        self.current_az = 0.0

        self.get_logger().info('Teleop active.')
        self.get_logger().info('E=Boot  Q=Shutdown  WASD=move')
        self.get_logger().info('R=Rebolar  B=Balance  P=Patinha')
        self.get_logger().info('C=TurnMode  X=OmniMode  Z=PoseMode  SPACE=Stop')

    def _get_key(self, settings):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.02)
        if rlist:
            key = sys.stdin.read(1)
            if key == '\x1b':
                extra = sys.stdin.read(2)
                key = key + extra
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def _pub_state(self, state: str):
        msg = String()
        msg.data = state
        self.state_pub.publish(msg)

    def _pub_vel(self, lx=0.0, ly=0.0, az=0.0):
        self.current_lx = lx
        self.current_ly = ly
        self.current_az = az
        msg = Twist()
        msg.linear.x  = lx
        msg.linear.y  = ly
        msg.angular.z = az
        self.vel_pub.publish(msg)

    def _republish_vel(self):
        msg = Twist()
        msg.linear.x  = self.current_lx
        msg.linear.y  = self.current_ly
        msg.angular.z = self.current_az
        self.vel_pub.publish(msg)

    def _stop(self):
        self._pub_vel()
        self._pub_state('IDLE')

    def run(self):
        settings = termios.tcgetattr(sys.stdin)
        try:
            while rclpy.ok():
                key = self._get_key(settings)

                if not key:
                    self._republish_vel()
                    continue

                if key == 'e':
                    self.get_logger().info('Booting...')
                    self._pub_state('BOOT')

                elif key == 'q':
                    self.get_logger().info('Shutting down...')
                    self._pub_state('SHUTDOWN')

                elif key == 'c':
                    self._pub_state('NAV_TURN')
                    self.get_logger().info('Nav mode: TURN')

                elif key == 'x':
                    self._pub_state('NAV_OMNI')
                    self.get_logger().info('Nav mode: OMNI')

                elif key == 'r':
                    self.get_logger().info('Rebolar')
                    self._pub_state('REBOLAR')

                elif key == 'b':
                    self.get_logger().info('Balance')
                    self._pub_state('BALANCE')

                elif key == 'p':
                    self.get_logger().info('Patinha')
                    self._pub_state('PATINHA')

                elif key == 'z':
                    self.pose_mode = not self.pose_mode
                    if not self.pose_mode:
                        self.pose_roll  = 0.0
                        self.pose_pitch = 0.0
                        self._pub_state('IDLE')
                    self.get_logger().info(f'Pose mode: {"ON" if self.pose_mode else "OFF"}')

                elif key == '\x1b[A':
                    if self.pose_mode:
                        self.pose_pitch = max(-POSE_MAX, self.pose_pitch - POSE_STEP)
                        self._pub_state(f'POSE {self.pose_roll:.1f} {self.pose_pitch:.1f}')
                    else:
                        self._pub_vel(lx= self.linear_speed)

                elif key == '\x1b[B':
                    if self.pose_mode:
                        self.pose_pitch = min( POSE_MAX, self.pose_pitch + POSE_STEP)
                        self._pub_state(f'POSE {self.pose_roll:.1f} {self.pose_pitch:.1f}')
                    else:
                        self._pub_vel(lx=-self.linear_speed)

                elif key == '\x1b[D':
                    if self.pose_mode:
                        self.pose_roll = max(-POSE_MAX, self.pose_roll - POSE_STEP)
                        self._pub_state(f'POSE {self.pose_roll:.1f} {self.pose_pitch:.1f}')
                    else:
                        self._pub_vel(az= self.angular_speed)

                elif key == '\x1b[C':
                    if self.pose_mode:
                        self.pose_roll = min( POSE_MAX, self.pose_roll + POSE_STEP)
                        self._pub_state(f'POSE {self.pose_roll:.1f} {self.pose_pitch:.1f}')
                    else:
                        self._pub_vel(az=-self.angular_speed)

                elif key == 'w':
                    self._pub_vel(lx= self.linear_speed)

                elif key == 's':
                    self._pub_vel(lx=-self.linear_speed)

                elif key == 'a':
                    self._pub_vel(az= self.angular_speed)

                elif key == 'd':
                    self._pub_vel(az=-self.angular_speed)

                elif key in (' ', '\x1b'):
                    self.pose_roll  = 0.0
                    self.pose_pitch = 0.0
                    self._stop()
                    self.get_logger().info('Stop')

                elif key == '\x03':
                    break

                else:
                    continue

                print(f'\r[KEY={repr(key)}]  pose=({self.pose_roll:.1f}°, {self.pose_pitch:.1f}°)',
                      end='', flush=True)

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            self._stop()
            print('\n[EXIT] Teleop terminated.')

def main(args=None):
    rclpy.init(args=args)
    node = TeleopHexapod()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
