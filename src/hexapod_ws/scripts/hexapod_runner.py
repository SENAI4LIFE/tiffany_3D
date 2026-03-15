#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu

import math
import numpy as np
import time

L1 = 0.0256
L2 = 0.0900
L3 = 0.1216

TOTAL_PONTOS          = 25
METADE_PONTOS         = TOTAL_PONTOS // 2
TOTAL_PONTOS_CIRCULAR = 25
STEP_LENGTH           = -0.080
GAIT_TICK             = 0.020

PATINHA_TOTAL  = 50
PATINHA_META   = PATINHA_TOTAL // 2
PATINHA_ROLL   = -10.0
PATINHA_PITCH  = -10.0
PATINHA_DX     = -0.100
PATINHA_DY     =  0.0
PATINHA_DZ     =  0.100

OFFSETS = [0, METADE_PONTOS, 0, METADE_PONTOS, 0, METADE_PONTOS]

SHOULDER_POSITIONS = [
    np.array([ 0.0930, -0.0555,  0.0]),
    np.array([ 0.0000, -0.0650,  0.0]),
    np.array([-0.0950, -0.0550,  0.0]),
    np.array([ 0.0930,  0.0555,  0.0]),
    np.array([ 0.0000,  0.0650,  0.0]),
    np.array([-0.0950,  0.0550,  0.0]),
]

ANGLES_STOW_BY_LEG = [
    ( 0.0,  90.0, -135.0),
    ( 0.0,  90.0, -135.0),
    ( 0.0,  90.0, -135.0),
    ( 0.0,  90.0, -135.0),
    ( 0.0,  90.0, -135.0),
    ( 0.0,  90.0, -135.0),
]

LEG_CONFIGS = [
    (-30.0,  25.0, -100.0, "right"),
    (  0.0,  25.0, -100.0, "right"),
    ( 30.0,  25.0, -100.0, "right"),
    ( 30.0,  25.0, -100.0, "left"),
    (  0.0,  25.0, -100.0, "left"),
    (-30.0,  25.0, -100.0, "left"),
]

def fk(ombro_deg: float, femur_deg: float, tibia_deg: float) -> np.ndarray:
    o = math.radians(ombro_deg)
    f = math.radians(femur_deg)
    t = math.radians(tibia_deg)
    x = -math.sin(o) * (L1 + L3 * math.cos(f + t) + L2 * math.cos(f))
    y =  math.cos(o) * (L1 + L3 * math.cos(f + t) + L2 * math.cos(f))
    z =  L3 * math.sin(f + t) + L2 * math.sin(f)
    return np.array([x, y, z])

def ik(xyz: np.ndarray):
    x, y, z   = float(xyz[0]), float(xyz[1]), float(xyz[2])
    y_prime   = math.sqrt(x*x + y*y) - L1
    Lv        = math.sqrt(z*z + y_prime*y_prime)
    cos_alpha = np.clip((L2**2 + L3**2 - Lv**2) / (2*L2*L3), -1.0, 1.0)
    cos_beta  = np.clip((Lv**2 + L2**2 - L3**2) / (2*Lv*L2), -1.0, 1.0)
    tibia_rad = -math.pi + math.acos(cos_alpha)
    ombro_rad = -math.atan2(x, y)
    femur_rad =  math.acos(cos_beta) + math.atan2(z, y_prime)
    return (ombro_rad, femur_rad, tibia_rad)

def rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    w = math.radians(yaw_deg)
    Rz = np.array([[math.cos(w), -math.sin(w), 0],
                   [math.sin(w),  math.cos(w), 0],
                   [0,            0,            1]])
    Ry = np.array([[ math.cos(p), 0, math.sin(p)],
                   [0,            1, 0           ],
                   [-math.sin(p), 0, math.cos(p)]])
    Rx = np.array([[1, 0,            0           ],
                   [0, math.cos(r), -math.sin(r)],
                   [0, math.sin(r),  math.cos(r)]])
    return Rz @ Ry @ Rx

def build_bezier_points(xyz_ini: np.ndarray):
    half = STEP_LENGTH / 2.0
    P0 = [xyz_ini[0] - half,           xyz_ini[2]]
    P1 = [P0[0] + half / 2.0,          P0[1] + 2.0 * abs(half)]
    P3 = [P0[0] + STEP_LENGTH,         P0[1]]
    P2 = [P3[0] - half / 2.0,          P0[1] + 2.0 * abs(half)]
    return P0, P1, P2, P3

def trajetoria_linear(xyz_ini, k, offset, angle_rad, P0, P1, P2, P3):
    kn = (k + offset) % TOTAL_PONTOS
    if kn < METADE_PONTOS:
        t  = float(kn) / (METADE_PONTOS - 1)
        u  = 1.0 - t
        bx = u**3*P0[0] + 3*u**2*t*P1[0] + 3*u*t**2*P2[0] + t**3*P3[0]
        bz = u**3*P0[1] + 3*u**2*t*P1[1] + 3*u*t**2*P2[1] + t**3*P3[1]
        dx = bx - xyz_ini[0]
        x  = xyz_ini[0] + math.cos(angle_rad) * dx
        y  = xyz_ini[1] + math.sin(angle_rad) * dx
        z  = bz
    else:
        t  = float(kn - METADE_PONTOS) / (METADE_PONTOS - 1)
        bx = P3[0] + (P0[0] - P3[0]) * t
        dx = bx - xyz_ini[0]
        x  = xyz_ini[0] + math.cos(angle_rad) * dx
        y  = xyz_ini[1] + math.sin(angle_rad) * dx
        z  = xyz_ini[2]
    return np.array([x, y, z])

def mapeia_circular(xyz_ini, xyz_atual, step_len, total_angle, shoulder):
    d_alpha = (total_angle / 2.0) * (xyz_atual[0] - xyz_ini[0]) / step_len
    x = xyz_ini[0] + shoulder[0]
    y = xyz_ini[1] + shoulder[1]
    R = math.sqrt(x*x + y*y)
    alpha   = math.atan2(x, y)
    n_alpha = alpha + d_alpha
    return np.array([R*math.sin(n_alpha) - shoulder[0],
                     R*math.cos(n_alpha) - shoulder[1],
                     xyz_atual[2]])

def bezier_pata(xyz_ini, k, dx, dy, dz, total):
    meta = total // 2
    kn   = k % total
    dx1, dx2 = dx / 4.0, dx / 2.0
    dy1, dy2 = dy / 4.0, dy / 2.0
    dz1, dz2 = dz / 4.0, dz / 2.0
    Px = [xyz_ini[0], xyz_ini[0]+dx1, xyz_ini[0]+dx2, xyz_ini[0]+dx]
    Py = [xyz_ini[1], xyz_ini[1]+dy1, xyz_ini[1]+dy2, xyz_ini[1]+dy]
    Pz = [xyz_ini[2], xyz_ini[2]+dz1+0.006, xyz_ini[2]+dz2+0.010, xyz_ini[2]+dz]
    if kn < meta:
        t = float(kn) / (meta - 1)
        u = 1.0 - t
        x = u**3*Px[0] + 3*u**2*t*Px[1] + 3*u*t**2*Px[2] + t**3*Px[3]
        y = u**3*Py[0] + 3*u**2*t*Py[1] + 3*u*t**2*Py[2] + t**3*Py[3]
        z = u**3*Pz[0] + 3*u**2*t*Pz[1] + 3*u*t**2*Pz[2] + t**3*Pz[3]
    else:
        x, y, z = Px[3], Py[3], Pz[3]
    return np.array([x, y, z])

def circular_roll_pitch_yaw(k, angle_max_deg):
    angle_max_rad = math.radians(angle_max_deg)
    kn        = k % TOTAL_PONTOS_CIRCULAR
    t         = float(kn) / TOTAL_PONTOS_CIRCULAR
    angle_rad = 2.0 * math.pi * t
    roll_deg  = math.cos(angle_rad) * math.degrees(angle_max_rad)
    pitch_deg = math.sin(angle_rad) * math.degrees(angle_max_rad)
    return roll_deg, pitch_deg, 0.0

def _rotacao_pata(ponto, roll_deg, pitch_deg, yaw_deg):
    r  = math.radians(roll_deg)
    pi = math.radians(pitch_deg)
    w  = math.radians(yaw_deg)
    x = (ponto[0]*math.cos(pi)*math.cos(w)
         + ponto[1]*(math.cos(w)*math.sin(pi)*math.sin(r) - math.cos(r)*math.sin(w))
         + ponto[2]*(math.sin(r)*math.sin(w) + math.cos(r)*math.cos(w)*math.sin(pi)))
    y = (ponto[0]*math.cos(pi)*math.sin(w)
         + ponto[1]*(math.cos(r)*math.cos(w) + math.sin(pi)*math.sin(r)*math.sin(w))
         + ponto[2]*(math.cos(r)*math.sin(pi)*math.sin(w) - math.cos(w)*math.sin(r)))
    z = (-ponto[0]*math.sin(pi)
         + ponto[1]*math.cos(pi)*math.sin(r)
         + ponto[2]*math.cos(pi)*math.cos(r))
    return np.array([x, y, z])

def lerp(a, b, t):
    return a + (b - a) * t

def compute_andar(k, angle_rad, xyz_ini, bezier):
    results = []
    for i in range(6):
        P0, P1, P2, P3 = bezier[i]
        current_angle   = angle_rad + (math.pi if i >= 3 else 0.0)
        xyz = trajetoria_linear(xyz_ini[i], k, OFFSETS[i], current_angle, P0, P1, P2, P3)
        results.append(ik(xyz))
    return results

def compute_andar_circular(k, angle_deg, xyz_ini, bezier):
    angle_abs = abs(angle_deg)
    angle_max = math.pi / 9.0
    if angle_deg < 0:
        angle_max = -angle_max
    v_mult = 1.0
    w_mult = 1.0
    if angle_abs in (0, 180):
        w_mult = 0.0
    elif angle_abs == 90:
        v_mult = 0.0

    dir_signs = [1, 1, 1, -1, -1, -1]
    results   = []
    for i in range(6):
        P0, P1, P2, P3 = bezier[i]
        step_len = P3[0] - xyz_ini[i][0]
        xyz_lin  = trajetoria_linear(xyz_ini[i], k, OFFSETS[i], 0, P0, P1, P2, P3)
        sign     = dir_signs[i]
        shoulder = SHOULDER_POSITIONS[i] * np.array([-1.0, sign, 1.0])
        xyz_rot  = mapeia_circular(xyz_ini[i], xyz_lin, step_len, angle_max * sign, shoulder)
        xyz_b    = (xyz_lin * v_mult + xyz_rot * w_mult) / (v_mult + w_mult)
        results.append(ik(xyz_b))
    return results

def compute_ik_corpo(roll_deg, pitch_deg, yaw_deg, xyz_ini):
    results  = []
    R        = rotation_matrix(-roll_deg, -pitch_deg, -yaw_deg)
    sig_list = [
        np.array([-1., -1.,  1.]),
        np.array([-1., -1.,  1.]),
        np.array([-1., -1.,  1.]),
        np.array([-1.,  1.,  1.]),
        np.array([-1.,  1.,  1.]),
        np.array([-1.,  1.,  1.]),
    ]
    for i in range(6):
        sig   = sig_list[i]
        ombro = SHOULDER_POSITIONS[i]
        ponto = xyz_ini[i] * sig + ombro
        rot   = R @ ponto
        xyz   = (rot - ombro) * sig
        results.append(ik(xyz))
    return results

def compute_rebolar(k, xyz_ini):
    roll_deg, pitch_deg, yaw_deg = circular_roll_pitch_yaw(k, 10)
    return compute_ik_corpo(roll_deg, pitch_deg, yaw_deg, xyz_ini)

def compute_dar_patinha(k, xyz_ini):
    t         = min(1.0, float(k) / max(1, PATINHA_META - 1))
    roll      = -PATINHA_ROLL  * t
    pitch     = -PATINHA_PITCH * t
    sig_right = np.array([-1.,  -1.,  1.])
    sig_left  = np.array([-1.,   1.,  1.])
    sig_list  = [sig_left, sig_left, sig_left, sig_right, sig_right, sig_right]
    results   = []
    for i in range(6):
        if i == 3:
            xyz = bezier_pata(xyz_ini[i], k, PATINHA_DX, PATINHA_DY, PATINHA_DZ, PATINHA_TOTAL)
        else:
            sig   = sig_list[i]
            ombro = SHOULDER_POSITIONS[i]
            ponto = xyz_ini[i] * sig + ombro
            rot   = _rotacao_pata(ponto, roll, pitch, 0.0)
            rotated = (rot - ombro) * sig
            xyz   = np.array([rotated[0], rotated[1], xyz_ini[i][2]])
        results.append(ik(xyz))
    return results

class HexapodRunner(Node):
    def __init__(self):
        super().__init__('hexapod_runner')

        self.joint_pub = self.create_publisher(
            Float64MultiArray, '/hexapod_controller/commands', 10)

        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_cb, 10)
        self.state_sub = self.create_subscription(
            String, '/tiffany/state', self._state_cb, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self._imu_cb, 10)

        self.xyz_ini, self.bezier = self._init_leg_state()

        self.state      = 'POWERED_OFF'
        self.prev_state = None

        self.k         = 0
        self.patinha_k = 0

        self.nav_mode = 'OMNI'

        self.pose_roll  = 0.0
        self.pose_pitch = 0.0
        self.POSE_MAX   = 15.0
        self.POSE_STEP  = 1.5

        self.angle_joystick = 0.0

        self.smoothed_rpy = [0.0, 0.0, 0.0]

        self.create_timer(0.02, self._step)
        self._publish_stow()
        self.get_logger().info('HexapodRunner ready. State: POWERED_OFF')
        self.get_logger().info('Send /tiffany/state = "BOOT" to start.')

    def _init_leg_state(self):
        xyz_ini = []
        bezier  = []
        for cfg in LEG_CONFIGS:
            coxa_h, femur_h, tibia_h, _ = cfg
            xyz = fk(coxa_h, femur_h, tibia_h)
            xyz_ini.append(xyz)
            bezier.append(build_bezier_points(xyz))
        return xyz_ini, bezier

    def _joints_from_results(self, results):

        flat = []
        for coxa_r, femur_r, tibia_r in results:
            flat.extend([coxa_r, -femur_r, -tibia_r])
        return flat

    def _publish_joints(self, results):
        msg = Float64MultiArray()
        msg.data = self._joints_from_results(results)
        self.joint_pub.publish(msg)

    def _publish_stow(self):
        results = [ik(fk(*ANGLES_STOW_BY_LEG[i])) for i in range(6)]
        self._publish_joints(results)

    def _publish_home(self):
        results = [ik(self.xyz_ini[i]) for i in range(6)]
        self._publish_joints(results)

    def _run_boot_sequence(self):
        steps = 50
        stow  = [fk(*ANGLES_STOW_BY_LEG[i]) for i in range(6)]

        for k in range(steps):
            t = float(k) / (steps - 1)
            results = []
            for i in range(6):
                xyz = np.array([lerp(stow[i][0], self.xyz_ini[i][0], t),
                                lerp(stow[i][1], self.xyz_ini[i][1], t),
                                stow[i][2]])
                results.append(ik(xyz))
            self._publish_joints(results)
            time.sleep(0.02)

        above = [np.array([self.xyz_ini[i][0], self.xyz_ini[i][1], stow[i][2]])
                 for i in range(6)]
        for k in range(steps):
            t = float(k) / (steps - 1)
            results = []
            for i in range(6):
                xyz = np.array([self.xyz_ini[i][0],
                                self.xyz_ini[i][1],
                                lerp(above[i][2], self.xyz_ini[i][2], t)])
                results.append(ik(xyz))
            self._publish_joints(results)
            time.sleep(0.02)

        self.k     = 0
        self.state = 'IDLE'
        self.get_logger().info('Boot complete. State: IDLE')

    def _run_shutdown_sequence(self):
        steps = 50
        stow  = [fk(*ANGLES_STOW_BY_LEG[i]) for i in range(6)]
        above = [np.array([self.xyz_ini[i][0], self.xyz_ini[i][1], stow[i][2]])
                 for i in range(6)]

        for k in range(steps):
            t = float(k) / (steps - 1)
            results = []
            for i in range(6):
                xyz = np.array([self.xyz_ini[i][0],
                                self.xyz_ini[i][1],
                                lerp(self.xyz_ini[i][2], above[i][2], t)])
                results.append(ik(xyz))
            self._publish_joints(results)
            time.sleep(0.02)

        for k in range(steps):
            t = float(k) / (steps - 1)
            results = []
            for i in range(6):
                xyz = np.array([lerp(self.xyz_ini[i][0], stow[i][0], t),
                                lerp(self.xyz_ini[i][1], stow[i][1], t),
                                above[i][2]])
                results.append(ik(xyz))
            self._publish_joints(results)
            time.sleep(0.02)

        self.state = 'POWERED_OFF'
        self.get_logger().info('Shutdown complete. State: POWERED_OFF')

    def _imu_cb(self, msg: Imu):

        q = msg.orientation

        sinr = 2.0 * (q.w * q.x + q.y * q.z)
        cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr, cosr)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = math.asin(np.clip(sinp, -1.0, 1.0))
        alpha = 0.2
        self.smoothed_rpy[0] += (roll  - self.smoothed_rpy[0]) * alpha
        self.smoothed_rpy[1] += (pitch - self.smoothed_rpy[1]) * alpha

    def _state_cb(self, msg: String):

        cmd = msg.data.upper().strip()

        if cmd == 'BOOT' and self.state == 'POWERED_OFF':
            self._run_boot_sequence()

        elif cmd == 'SHUTDOWN' and self.state != 'POWERED_OFF':
            self._run_shutdown_sequence()

        elif cmd in ('IDLE', 'BALANCE', 'REBOLAR', 'PATINHA') and self.state != 'POWERED_OFF':
            if cmd == 'PATINHA':
                if self.state == 'PATINHA':
                    self.state = 'IDLE'
                    self.get_logger().info('State → IDLE')
                    return
                self.patinha_k = 0
            self.state = cmd
            self.get_logger().info(f'State → {cmd}')

        elif cmd == 'NAV_OMNI':
            self.nav_mode = 'OMNI'
        elif cmd == 'NAV_TURN':
            self.nav_mode = 'TURN'

        elif cmd.startswith('POSE '):

            try:
                _, r, p = cmd.split()
                self.pose_roll  = float(r)
                self.pose_pitch = float(p)
                self.state = 'POSE'
            except ValueError:
                pass

    def _cmd_vel_cb(self, msg: Twist):

        if self.state == 'POWERED_OFF':
            return

        lx = msg.linear.x
        ly = msg.linear.y
        az = msg.angular.z

        if abs(lx) > 0.01 or abs(ly) > 0.01 or abs(az) > 0.01:
            if abs(lx) > 0.01:
                self.angle_joystick = 180.0 if lx > 0 else 0.0
            elif abs(ly) > 0.01:
                self.angle_joystick = -90.0 if ly > 0 else 90.0
            elif abs(az) > 0.01:
                self.angle_joystick = -90.0 if az > 0 else 90.0

            self.state = 'WALKING' if self.nav_mode == 'OMNI' else 'TURNING'
        else:
            if self.state in ('WALKING', 'TURNING'):
                self.state = 'IDLE'



    def _step(self):

        state = self.state

        if state == 'WALKING':
            angle_rad = math.radians(self.angle_joystick)
            results   = compute_andar(self.k, angle_rad, self.xyz_ini, self.bezier)
            self._publish_joints(results)
            self.k = (self.k + 1) % TOTAL_PONTOS

        elif state == 'TURNING':
            results = compute_andar_circular(
                self.k, self.angle_joystick, self.xyz_ini, self.bezier)
            self._publish_joints(results)
            if abs(self.angle_joystick) > 90:
                self.k = (self.k - 1) % TOTAL_PONTOS
            else:
                self.k = (self.k + 1) % TOTAL_PONTOS

        elif state == 'BALANCE':
            roll_deg  = math.degrees(self.smoothed_rpy[0])
            pitch_deg = math.degrees(self.smoothed_rpy[1])
            results   = compute_ik_corpo(roll_deg, pitch_deg, 0.0, self.xyz_ini)
            self._publish_joints(results)

        elif state == 'POSE':
            results = compute_ik_corpo(
                self.pose_roll, self.pose_pitch, 0.0, self.xyz_ini)
            self._publish_joints(results)

        elif state == 'PATINHA':
            results = compute_dar_patinha(self.patinha_k, self.xyz_ini)
            self._publish_joints(results)
            if self.patinha_k < PATINHA_META - 1:
                self.patinha_k += 1

        elif state == 'REBOLAR':
            results = compute_rebolar(self.k, self.xyz_ini)
            self._publish_joints(results)
            self.k = (self.k + 1) % TOTAL_PONTOS_CIRCULAR

        elif state == 'IDLE':
            if self.prev_state != 'IDLE':
                self._publish_home()

        self.prev_state = state

def main(args=None):
    rclpy.init(args=args)
    node = HexapodRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
