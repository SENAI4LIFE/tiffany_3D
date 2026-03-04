import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import bootcamp

SIM_FREQ = 120
DT = 1.0 / SIM_FREQ

STOWED_FEMUR = -80.0
STOWED_TIBIA  = 140.0
PEAK_FEMUR    = -60.0
PEAK_TIBIA    = 120.0
STEP_HEIGHT   = 0.12
STEP_STRIDE   = 20.0

COXA_LIMIT  = 40.0
FEMUR_MIN   = -85.0
FEMUR_MAX   =  30.0
TIBIA_MIN   =  30.0
TIBIA_MAX   = 150.0

LEG_CONFIGS = [
    (0,  1,  2,  -30, -26, 100, "right"),
    (3,  4,  5,    0, -26, 100, "right"),
    (6,  7,  8,   30, -26, 100, "right"),
    (9,  10, 11,  30, -26, 100, "left"),
    (12, 13, 14,   0, -26, 100, "left"),
    (15, 16, 17, -30, -26, 100, "left"),
]

GAIT_PHASES = [0, math.pi, 0, math.pi, 0, math.pi]

_TWO_PI = 2.0 * math.pi
_PI     = math.pi
_RAD    = math.pi / 180.0
_DEG    = 180.0 / math.pi

BALANCE_ROLL_GAIN  = 12.0
BALANCE_PITCH_GAIN = 12.0
BALANCE_DAMP       = 0.20


def setup_simulation():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
    p.resetDebugVisualizerCamera(5.0, 270, -25, [-5.0, 0, 5.0])
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(
        numSolverIterations=8,
        numSubSteps=1,
        fixedTimeStep=DT,
        enableConeFriction=0,
    )
    p.loadURDF("plane.urdf", [0, 0, 0])
    bootcamp.CreateTerrain()
    robot = p.loadURDF("robot.urdf", [0, 0, 0])
    for j in range(p.getNumJoints(robot)):
        p.changeDynamics(robot, j,
                         lateralFriction=5.0,
                         jointDamping=0.1,
                         restitution=0.0,
                         maxJointVelocity=6.0)
    return robot


def calibrate_segments(robot):
    tibia_info = p.getJointInfo(robot, 2)
    L1 = float(np.linalg.norm(tibia_info[14]))
    L2 = 0.15
    for j in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, j)
        if info[16] == 2:
            L2 = float(np.linalg.norm(info[14]))
            break
    return L1, L2


def fk_2d(femur_deg, tibia_deg, L1, L2):
    f  = femur_deg * _RAD
    ft = f + tibia_deg * _RAD
    return (L1 * math.cos(f)  + L2 * math.cos(ft),
            L1 * math.sin(f)  + L2 * math.sin(ft))


def ik_2d(ext, dep, L1, L2):
    r_sq  = ext * ext + dep * dep
    max_r = L1 + L2
    if r_sq > max_r * max_r:
        s   = max_r / math.sqrt(r_sq)
        ext *= s;  dep *= s
        r_sq = max_r * max_r
    cos_t = (r_sq - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    cos_t = 1.0 if cos_t > 1.0 else (-1.0 if cos_t < -1.0 else cos_t)
    t     = math.acos(cos_t)
    sin_t = math.sin(t)
    f     = math.atan2(dep, ext) - math.atan2(L2 * sin_t, L1 + L2 * cos_t)
    femur_deg = f * _DEG
    tibia_deg = t * _DEG
    femur_deg = max(FEMUR_MIN, min(FEMUR_MAX, femur_deg))
    tibia_deg = max(TIBIA_MIN, min(TIBIA_MAX, tibia_deg))
    return femur_deg, tibia_deg


def clamp_coxa(coxa_home, deviation):
    clamped = max(-COXA_LIMIT, min(COXA_LIMIT, deviation))
    return coxa_home + clamped, clamped


def bezier3(p0, p1, p2, p3, t):
    u = 1.0 - t
    return u*u*u*p0 + 3.0*u*u*t*p1 + 3.0*u*t*t*p2 + t*t*t*p3


def set_leg(robot, coxa_id, femur_id, tibia_id, coxa_deg, femur_deg, tibia_deg):
    mv = 4.0
    p.setJointMotorControl2(robot, coxa_id,  p.POSITION_CONTROL, coxa_deg  * _RAD, maxVelocity=mv)
    p.setJointMotorControl2(robot, femur_id, p.POSITION_CONTROL, femur_deg * _RAD, maxVelocity=mv)
    p.setJointMotorControl2(robot, tibia_id, p.POSITION_CONTROL, tibia_deg * _RAD, maxVelocity=mv)


def get_body_tilt(robot, smoothed_rpy):
    _, orn = p.getBasePositionAndOrientation(robot)
    rpy    = p.getEulerFromQuaternion(orn)
    for i in range(3):
        smoothed_rpy[i] += (rpy[i] - smoothed_rpy[i]) * BALANCE_DAMP
    return smoothed_rpy[0], smoothed_rpy[1]


def compute_balance_dep_offset(leg_id, roll, pitch):
    x_sign = ( 1, 0, -1,  1, 0, -1)[leg_id]
    y_sign = (-1,-1, -1,  1, 1,  1)[leg_id]
    return pitch * BALANCE_PITCH_GAIN * x_sign + roll * BALANCE_ROLL_GAIN * y_sign


def compute_powered_on_angles(leg_id, cfg, gait_phase, fwd, lat, nav_mode,
                               rest_coeff, coxa_mem, radial_mem, height_mem,
                               L1, L2, roll, pitch):
    _, _, _, coxa_home, femur_home, tibia_home, side = cfg
    ext_n, dep_n = fk_2d(femur_home, tibia_home, L1, L2)
    phase  = gait_phase % _TWO_PI
    moving = abs(fwd) > 0.01 or abs(lat) > 0.01

    balance_off = compute_balance_dep_offset(leg_id, roll, pitch)

    if moving:
        spd = math.sqrt(fwd * fwd + lat * lat)
        mag = spd * 2.0
        if mag > 1.0: mag = 1.0
        lift = STEP_HEIGHT * mag

        if phase < _PI:
            swing_t = phase / _PI
            dep = bezier3(dep_n, dep_n - lift * 1.3, dep_n - lift * 1.3, dep_n, swing_t) if swing_t > 0.0 else dep_n
        else:
            dep = dep_n + balance_off * 0.012

        osc = math.cos(phase) * STEP_STRIDE
        
        if nav_mode == "OMNI":
            coxa_cmd  = osc * fwd
            radial_cmd = (-osc * lat * 0.003) if side == "right" else (osc * lat * 0.003)
        else:
            fwd_c  = osc * fwd
            turn_c = osc * lat if side == "right" else -osc * lat
            coxa_cmd   = fwd_c + turn_c
            radial_cmd = osc * fwd * 0.0025
            if leg_id in (1, 4):
                radial_cmd = 0.0
            elif leg_id in (2, 5):
                radial_cmd = -radial_cmd

        coxa_mem[leg_id]   += (coxa_cmd   - coxa_mem[leg_id])   * 0.2
        radial_mem[leg_id] += (radial_cmd - radial_mem[leg_id]) * 0.2
        height_mem[leg_id]  = dep

        coxa_out, clamped_dev = clamp_coxa(coxa_home, coxa_mem[leg_id])
        coxa_mem[leg_id] = clamped_dev
        if side == "right":
            coxa_out = coxa_home - clamped_dev

        arc = ext_n / math.cos(clamped_dev * _RAD)
        femur_out, tibia_out = ik_2d(arc + radial_mem[leg_id], dep, L1, L2)

    else:
        w  = rest_coeff
        w1 = 1.0 - w
        c  = coxa_mem[leg_id]   * w1
        r  = radial_mem[leg_id] * w1
        d  = height_mem[leg_id] * w1 + (dep_n + balance_off * 0.012) * w
        c  = max(-COXA_LIMIT, min(COXA_LIMIT, c))

        arc      = ext_n / math.cos(c * _RAD)
        coxa_out = coxa_home + (c if side == "left" else -c)
        femur_out, tibia_out = ik_2d(arc + r, d, L1, L2)
        if w < 1.0:
            coxa_mem[leg_id]   = c
            radial_mem[leg_id] = r

    return coxa_out, femur_out, tibia_out


def compute_transition_angles(cfg, state, timer):
    _, _, _, coxa_home, femur_home, tibia_home, _ = cfg
    coxa = femur = tibia = 0.0
    inv  = (4.0 - timer) if state == "POWERING_OFF" else timer

    if state == "WORLD_TO_POWERED_OFF_TRANS":
        a = timer * 0.5
        femur, tibia = STOWED_FEMUR * a, STOWED_TIBIA * a
    elif state == "POWERING_ON_TO_WORLD_TRANS":
        a = 1.0 - timer * 0.5
        femur, tibia = STOWED_FEMUR * a, STOWED_TIBIA * a
    elif inv <= 1.0:
        femur, tibia = STOWED_FEMUR, STOWED_TIBIA * (1.0 - inv) + tibia_home * inv
    elif inv <= 2.0:
        a = inv - 1.0
        femur = STOWED_FEMUR + (PEAK_FEMUR - STOWED_FEMUR) * a
        tibia = tibia_home
    elif inv <= 3.0:
        a = inv - 2.0
        coxa, femur, tibia = coxa_home * a, PEAK_FEMUR, tibia_home
    else:
        a = inv - 3.0
        coxa  = coxa_home
        femur = PEAK_FEMUR + (femur_home - PEAK_FEMUR) * a
        tibia = tibia_home

    return coxa, femur, tibia


def main():
    robot       = setup_simulation()
    L1, L2      = calibrate_segments(robot)
    _, dep_neutral = fk_2d(-26, 100, L1, L2)

    state               = "WORLD"
    trans_timer         = 0.0
    boot_clock          = 0.0
    stand_blend         = 0.0
    gait_acc            = 0.0
    last_key            = None
    revert_after_shutdown = False
    nav_mode            = "TURN"
    cam_track           = False
    vel_fwd             = 0.0
    vel_lat             = 0.0
    rest_coeff          = 1.0
    smoothed_rpy        = [0.0, 0.0, 0.0]

    leg_phases  = list(GAIT_PHASES)
    coxa_mem    = [0.0] * 6
    radial_mem  = [0.0] * 6
    height_mem  = [dep_neutral] * 6

    TRANS_STATES = frozenset((
        "POWERED_OFF_TO_POWERED_ON_TRANS", "POWERING_OFF",
        "WORLD_TO_POWERED_OFF_TRANS",      "POWERING_ON_TO_WORLD_TRANS",
    ))

    KEY_F, KEY_C, KEY_X = ord('f'), ord('c'), ord('x')
    KEY_Q, KEY_E, KEY_R = ord('q'), ord('e'), ord('r')

    sleep    = time.sleep
    get_keys = p.getKeyboardEvents
    step_sim = p.stepSimulation

    UP    = p.B3G_UP_ARROW
    DOWN  = p.B3G_DOWN_ARROW
    LEFT  = p.B3G_LEFT_ARROW
    RIGHT = p.B3G_RIGHT_ARROW
    ARROW_KEYS = (UP, DOWN, LEFT, RIGHT)

    active_input_axis = None

    while True:
        keys = get_keys()

        if keys.get(KEY_F, 0) & p.KEY_WAS_TRIGGERED: cam_track = not cam_track
        if keys.get(KEY_C, 0) & p.KEY_WAS_TRIGGERED: nav_mode  = "TURN"
        if keys.get(KEY_X, 0) & p.KEY_WAS_TRIGGERED: nav_mode  = "OMNI"

        if cam_track:
            pos, _ = p.getBasePositionAndOrientation(robot)
            p.resetDebugVisualizerCamera(5.0, 270, -25, [pos[0] - 5.0, pos[1], pos[2] + 5.0])

        if last_key and not (last_key in keys and keys[last_key] & p.KEY_IS_DOWN):
            last_key = None
        if not last_key:
            for k in keys:
                if keys[k] & p.KEY_IS_DOWN:
                    last_key = k
                    break

        in_transition = state.endswith("_TRANS") or state.startswith("POWERING")
        in_fwd = in_lat = 0

        if not in_transition:
            if last_key == KEY_Q:
                if   state == "WORLD":       state, trans_timer, revert_after_shutdown = "WORLD_TO_POWERED_OFF_TRANS", 0.0, False
                elif state == "POWERED_OFF": state, trans_timer = "POWERING_ON_TO_WORLD_TRANS", 0.0
                elif state == "POWERED_ON":  state, trans_timer, revert_after_shutdown = "POWERING_OFF", 0.0, True
            elif last_key == KEY_E:
                if   state == "WORLD":       state, boot_clock, stand_blend = "POWERING_ON_SEQUENTIAL", 0.0, 0.0
                elif state == "POWERED_OFF": state, trans_timer = "POWERED_OFF_TO_POWERED_ON_TRANS", 0.0
            elif last_key == KEY_R:
                if   state == "POWERED_ON": state, trans_timer, revert_after_shutdown = "POWERING_OFF", 0.0, False
                elif state == "WORLD":      state, trans_timer = "WORLD_TO_POWERED_OFF_TRANS", 0.0

        fwd_pressed = (UP   in keys and keys[UP]   & p.KEY_IS_DOWN) or \
                      (DOWN in keys and keys[DOWN]  & p.KEY_IS_DOWN)
        lat_pressed = (LEFT  in keys and keys[LEFT]  & p.KEY_IS_DOWN) or \
                      (RIGHT in keys and keys[RIGHT] & p.KEY_IS_DOWN)

        if fwd_pressed and not lat_pressed:
            active_input_axis = "fwd"
        elif lat_pressed and not fwd_pressed:
            active_input_axis = "lat"
        elif not fwd_pressed and not lat_pressed:
            active_input_axis = None

        dir_keys = [k for k in ARROW_KEYS if k in keys and keys[k] & p.KEY_IS_DOWN]

        if state == "POWERED_ON":
            for i in range(6):
                err = (GAIT_PHASES[i] - leg_phases[i] + _PI) % _TWO_PI - _PI
                leg_phases[i] += err * 0.05

            if dir_keys:
                gait_acc   += 0.025
                rest_coeff  = 0.0
                if active_input_axis == "fwd":
                    in_fwd = -1 if UP   in dir_keys else 1
                    in_lat = 0
                elif active_input_axis == "lat":
                    in_lat = -1 if RIGHT in dir_keys else 1
                    in_fwd = 0
            else:
                rest_coeff = min(1.0, rest_coeff + 0.02)
        else:
            vel_fwd = vel_lat = gait_acc = 0.0
            rest_coeff = 1.0

        vel_fwd += (in_fwd - vel_fwd) * 0.1
        vel_lat += (in_lat - vel_lat) * 0.1

        if state == "POWERING_ON_TO_WORLD_TRANS":
            trans_timer += 0.01
            if trans_timer >= 2.0: state = "WORLD"
        elif state == "WORLD_TO_POWERED_OFF_TRANS":
            trans_timer += 0.01
            if trans_timer >= 2.0: state = "POWERED_OFF"
        elif state == "POWERING_OFF":
            trans_timer += 0.01
            if trans_timer >= 4.0:
                state = "POWERING_ON_TO_WORLD_TRANS" if revert_after_shutdown else "POWERED_OFF"
                trans_timer = 0.0
        elif state == "POWERED_OFF_TO_POWERED_ON_TRANS":
            trans_timer += 0.01
            if trans_timer >= 4.0: state = "POWERED_ON"
        elif state == "POWERING_ON_SEQUENTIAL":
            boot_clock = min(7.0, boot_clock + 0.03)
            if boot_clock >= 7.0:
                stand_blend = min(1.0, stand_blend + 0.005)
                if stand_blend >= 1.0: state = "POWERED_ON"

        is_powered_on  = state == "POWERED_ON"
        is_powered_off = state == "POWERED_OFF"
        is_world       = state == "WORLD"
        is_sequential  = state == "POWERING_ON_SEQUENTIAL"
        is_trans       = state in TRANS_STATES

        roll = pitch = 0.0
        if is_powered_on:
            roll, pitch = get_body_tilt(robot, smoothed_rpy)

        for leg_id, cfg in enumerate(LEG_CONFIGS):
            coxa_id, femur_id, tibia_id, coxa_home, femur_home, tibia_home, side = cfg

            if is_world:
                coxa = femur = tibia = 0.0

            elif is_powered_off:
                coxa, femur, tibia = 0.0, STOWED_FEMUR, STOWED_TIBIA

            elif is_sequential:
                coxa  = coxa_home * max(0.0, min(1.0, boot_clock))
                w     = max(0.0, min(1.0, (boot_clock - 1.0) - leg_id))
                femur = (PEAK_FEMUR * (1.0 - stand_blend) + femur_home * stand_blend) * w
                tibia = ((PEAK_TIBIA - 20.0) * (1.0 - stand_blend) + tibia_home * stand_blend) * w

            elif is_trans:
                coxa, femur, tibia = compute_transition_angles(cfg, state, trans_timer)

            elif is_powered_on:
                phase = gait_acc + leg_phases[leg_id]
                coxa, femur, tibia = compute_powered_on_angles(
                    leg_id, cfg, phase, vel_fwd, vel_lat, nav_mode,
                    rest_coeff, coxa_mem, radial_mem, height_mem,
                    L1, L2, roll, pitch)
            else:
                coxa = femur = tibia = 0.0

            set_leg(robot, coxa_id, femur_id, tibia_id, coxa, femur, tibia)

        step_sim()
        sleep(DT)


main()