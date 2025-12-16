import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import style

style.use("ggplot")

# -------------------- WORLD --------------------
GAME_WIDTH  = 15
GAME_LENGTH = 15

SURVIVE_GOAL_STEPS = 300
TOT_EPISODES = 15000

# -------------------- REWARDS -------------------
HIT_PENALTY     = 120
STEP_REWARD     = 2
SURVIVE_REWARD  = 200
IDLE_PENALTY    = 1

# -------------------- EXPLORATION ---------------
epsilon   = 1.0
EPS_DECAY = 0.9997
EPS_MIN   = 0.05
SHOW_EVERY = 1000

# -------------------- Q-LEARNING ----------------
LR = 0.1
DISCOUNT = 0.95

# -------------------- RENDER --------------------
AGENT  = 1
BULLET = 2
d = {
    1: (255, 0, 0),
    2: (192, 192, 192),
}

# -------------------- ACTIONS -------------------
# We allow the agent to move faster than bullets:
# 0: up2, 1: down2, 2: left2, 3: right2, 4: stay
ACTIONS = 5
AGENT_STEP = 2       # player moves 2 cells per step
BULLET_STEP = 1      # bullet moves 1 cell per step

# -------------------- STATE ---------------------
SECTORS = 4
TIME_BINS = 10
DANGER_NEAR = 2
DANGER_MID  = 5

# -------------------- FIXED SPAWN ---------------
SPAWN_P_FIXED = 0.20   # fixed per-step spawn probability


class agent:
    def __init__(self):
        self.x = GAME_WIDTH // 2
        self.y = GAME_LENGTH // 2

    def action(self, choice: int):
        if choice == 0:
            self.move(0, -AGENT_STEP)
        elif choice == 1:
            self.move(0,  AGENT_STEP)
        elif choice == 2:
            self.move(-AGENT_STEP, 0)
        elif choice == 3:
            self.move( AGENT_STEP, 0)
        elif choice == 4:
            self.move(0, 0)

    def move(self, dx: int, dy: int):
        self.x += dx
        self.y += dy
        if self.x < 0: self.x = 0
        if self.x > GAME_WIDTH - 1: self.x = GAME_WIDTH - 1
        if self.y < 0: self.y = 0
        if self.y > GAME_LENGTH - 1: self.y = GAME_LENGTH - 1


class bullet_hell:
    """
    Straight-line bullets:
    - Spawn at border
    - Choose a fixed velocity (vx, vy) at spawn time
    - Then move in a fixed direction; DO NOT track agent
    """
    def __init__(self):
        # each bullet: [x, y, vx, vy]
        self.bullets = []

    def reset(self):
        self.bullets = []

    @staticmethod
    def _sign(z: int) -> int:
        if z > 0: return 1
        if z < 0: return -1
        return 0

    def _spawn_one(self):
        # border spawn
        side = np.random.randint(0, 4)
        if side == 0:  # top
            x = np.random.randint(0, GAME_WIDTH);  y = 0
        elif side == 1:  # bottom
            x = np.random.randint(0, GAME_WIDTH);  y = GAME_LENGTH - 1
        elif side == 2:  # left
            x = 0;  y = np.random.randint(0, GAME_LENGTH)
        else:  # right
            x = GAME_WIDTH - 1;  y = np.random.randint(0, GAME_LENGTH)

        # choose a fixed target point inside the map (not the agent!)
        tx = np.random.randint(0, GAME_WIDTH)
        ty = np.random.randint(0, GAME_LENGTH)

        vx = self._sign(tx - x)
        vy = self._sign(ty - y)

        # ensure it actually moves
        if vx == 0 and vy == 0:
            vx = 1

        return [x, y, vx, vy]

    def step(self):
        # fixed spawn
        if np.random.random() < SPAWN_P_FIXED:
            self.bullets.append(self._spawn_one())

        # move bullets in fixed direction
        for b in self.bullets:
            b[0] += BULLET_STEP * b[2]
            b[1] += BULLET_STEP * b[3]

        # keep bullets that are still in bounds
        self.bullets = [
            b for b in self.bullets
            if (0 <= b[0] < GAME_WIDTH) and (0 <= b[1] < GAME_LENGTH)
        ]

    def hit(self, ax: int, ay: int) -> bool:
        for b in self.bullets:
            if b[0] == ax and b[1] == ay:
                return True
        return False


def _sector_index(dx: int, dy: int) -> int:
    # 4 sectors: Right(0), Up(1), Left(2), Down(3)
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) >= abs(dy):
        return 0 if dx > 0 else 2
    else:
        return 3 if dy > 0 else 1


def get_state(player: agent, env_bullets: bullet_hell, step_i: int):
    # min manhattan dist per sector
    min_dist = [10**9] * SECTORS

    for b in env_bullets.bullets:
        dx = b[0] - player.x
        dy = b[1] - player.y
        s = _sector_index(dx, dy)
        dist = abs(dx) + abs(dy)
        if dist < min_dist[s]:
            min_dist[s] = dist

    danger = []
    for dist in min_dist:
        if dist <= DANGER_NEAR:
            danger.append(2)
        elif dist <= DANGER_MID:
            danger.append(1)
        else:
            danger.append(0)

    near_wall = 1 if (player.x == 0 or player.x == GAME_WIDTH-1 or player.y == 0 or player.y == GAME_LENGTH-1) else 0
    time_bin = min(TIME_BINS - 1, int(TIME_BINS * step_i / max(1, SURVIVE_GOAL_STEPS)))

    return tuple(danger + [near_wall, time_bin])


# -------------------- Q TABLE --------------------
q_table = {}

def _all_states_iter():
    levels = [0, 1, 2]
    for a0 in levels:
        for a1 in levels:
            for a2 in levels:
                for a3 in levels:
                    for nw in [0, 1]:
                        for tb in range(TIME_BINS):
                            yield (a0, a1, a2, a3, nw, tb)

for s in _all_states_iter():
    q_table[s] = [np.random.uniform(-1.0, 0.0) for _ in range(ACTIONS)]
