import pybullet as p
import numpy as np

# Global Configuration
DIFFICULTY  = 3.0
DIFF_MIN    = 1.0
DIFF_MAX    = 5.0
PROP_SIZE   = 10.0
START_X     = 10.0
NUM_PROPS   = 10
TERRAIN_END_X = START_X + NUM_PROPS * PROP_SIZE

def create_block(x, y, w, l, h, cid):
    sh = p.createCollisionShape(p.GEOM_BOX, halfExtents=[l/2, w/2, h/2], physicsClientId=cid)
    p.createMultiBody(0, sh, basePosition=[x, y, h/2], physicsClientId=cid)

def create_slope(x, y, w, l, h, cid):
    verts = [[-l/2,-w/2,0],[l/2,-w/2,0],[l/2,w/2,0],[-l/2,w/2,0],[l/2,-w/2,h],[l/2,w/2,h]]
    idx   = [0,2,1, 0,3,2, 0,1,4, 1,2,4, 2,5,4, 2,3,5, 3,0,5, 0,4,5]
    sh    = p.createCollisionShape(p.GEOM_MESH, vertices=verts, indices=idx, physicsClientId=cid)
    p.createMultiBody(0, sh, basePosition=[x, y, 0], physicsClientId=cid)

def create_stepping(x, y, w, l, h, cid):
    divs = 4
    sl, sw = l/divs, w/divs
    for r in range(divs):
        for c in range(divs):
            ch  = np.random.uniform(h/3, h)
            px  = (x - l/2) + (c * sl) + (sl/2)
            py  = (y - w/2) + (r * sw) + (sw/2)
            sh  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sl/2, sw/2, ch/2], physicsClientId=cid)
            p.createMultiBody(0, sh, basePosition=[px, py, ch/2], physicsClientId=cid)

def create_stairs(x, y, w, l, h, cid):
    steps = 5
    s_len = l / steps
    for i in range(steps):
        ch  = (h / steps) * (i + 1)
        px  = (x - l/2) + (i * s_len) + (s_len/2)
        sh  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s_len/2, w/2, ch/2], physicsClientId=cid)
        p.createMultiBody(0, sh, basePosition=[px, y, ch/2], physicsClientId=cid)

def create_roundy(x, y, w, l, h, cid):
    radius = (l**2 + 4*h**2) / (8*h)
    sh     = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=cid)
    p.createMultiBody(0, sh, basePosition=[x, y, h - radius], physicsClientId=cid)

def create_pyramid(x, y, w, l, h, cid):
    verts = [[-l/2,-w/2,0],[l/2,-w/2,0],[l/2,w/2,0],[-l/2,w/2,0],[0,0,h]]
    idx   = [0,2,1, 0,3,2, 0,1,4, 1,2,4, 2,3,4, 3,0,4]
    sh    = p.createCollisionShape(p.GEOM_MESH, vertices=verts, indices=idx, physicsClientId=cid)
    p.createMultiBody(0, sh, basePosition=[x, y, 0], physicsClientId=cid)

def create_logs(x, y, w, l, h, cid):
    num, r  = 4, h/2
    spacing = l/num
    sh      = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=w, physicsClientId=cid)
    for i in range(num):
        px = (x - l/2) + (i * spacing) + (spacing/2)
        p.createMultiBody(0, sh,
                          baseOrientation=p.getQuaternionFromEuler([1.57,0,0]),
                          basePosition=[px, y, r], physicsClientId=cid)

def create_slant(x, y, w, l, h, cid):
    verts = [[-l/2,-w/2,0],[l/2,-w/2,0],[l/2,w/2,0],[-l/2,w/2,0],[-l/2,w/2,h],[l/2,w/2,h]]
    idx   = [0,2,1, 0,3,2, 3,4,5, 3,5,2, 0,1,5, 0,5,4, 0,4,3, 1,2,5]
    sh    = p.createCollisionShape(p.GEOM_MESH, vertices=verts, indices=idx, physicsClientId=cid)
    p.createMultiBody(0, sh, basePosition=[x, y, 0], physicsClientId=cid)

def create_beams(x, y, w, l, h, cid):
    num_beams = 6
    bw = w / num_beams
    for i in range(num_beams):
        bh  = h if i % 2 == 0 else h * 0.5
        py  = (y - w/2) + (i * bw) + (bw/2)
        sh  = p.createCollisionShape(p.GEOM_BOX, halfExtents=[l/2, bw/2, bh/2], physicsClientId=cid)
        p.createMultiBody(0, sh, basePosition=[x, py, bh/2], physicsClientId=cid)

def create_canyon(x, y, w, l, h, cid):
    gap_ratio  = min(0.15 * DIFFICULTY, 0.4)
    gap        = l * gap_ratio
    platform_l = (l - gap) / 2
    sh = p.createCollisionShape(p.GEOM_BOX, halfExtents=[platform_l/2, w/2, h/2], physicsClientId=cid)
    p.createMultiBody(0, sh, basePosition=[x - gap/2 - platform_l/2, y, h/2], physicsClientId=cid)
    p.createMultiBody(0, sh, basePosition=[x + gap/2 + platform_l/2, y, h/2], physicsClientId=cid)

def CreateTerrain(client_id=0, lane_y=0.0):
    terrain_map = {
        "slope": create_slope, "block": create_block, "stepping": create_stepping,
        "stairs": create_stairs, "roundy": create_roundy, "pyramid": create_pyramid,
        "logs": create_logs, "slant": create_slant, "beams": create_beams, "canyon": create_canyon,
    }
    props = [
        {"type": "slope", "h": 0.25}, 
        {"type": "block", "h": 0.15}, 
        {"type": "stepping", "h": 0.35},
        {"type": "stairs", "h": 0.40}, 
        {"type": "roundy", "h": 0.60}, 
        {"type": "pyramid", "h": 0.55},
        {"type": "logs", "h": 0.30}, 
        {"type": "slant", "h": 0.45}, 
        {"type": "beams", "h": 0.40},
        {"type": "canyon", "h": 0.30},
    ]
    for i, cfg in enumerate(props):
        cx = START_X + (i * PROP_SIZE)
        sh = cfg["h"] * DIFFICULTY
        terrain_map[cfg["type"]](cx, lane_y, PROP_SIZE, PROP_SIZE, sh, client_id)
