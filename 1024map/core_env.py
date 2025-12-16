# core_env.py
import math
import random
from dataclasses import dataclass

W, H = 1024, 1024

@dataclass
class Player:
    x: float
    y: float
    r: float = 14.0
    speed: float = 320.0  # px/s

@dataclass
class Bullet:
    x: float
    y: float
    vx: float
    vy: float
    r: float = 8.0

class BulletHellEnv:
    """
    Continuous 2D bullet-hell environment (no pygame dependency).
    - Bullets spawn from borders at fixed interval, move straight with fixed velocity (no homing).
    - Player moves with discrete actions.
    """
    def __init__(
        self,
        spawn_interval: float = 0.12,   # seconds per bullet
        bullet_speed: float = 220.0,    # px/s
        max_bullets: int = 800,
        survival_seconds: float = 20.0,
        idle_penalty: float = 0.9,
    ):
        self.spawn_interval = float(spawn_interval)
        self.bullet_speed = float(bullet_speed)
        self.max_bullets = int(max_bullets)
        self.survival_seconds = float(survival_seconds)
        self.idle_penalty = float(idle_penalty)

        self.player = Player(W / 2, H / 2)
        self.bullets: list[Bullet] = []

        self.t = 0.0
        self._spawn_acc = 0.0
        self.done = False

    def reset(self):
        self.player = Player(W / 2, H / 2)
        self.bullets = []
        self.t = 0.0
        self._spawn_acc = 0.0
        self.done = False

    def _clamp_player(self):
        p = self.player
        p.x = max(p.r, min(W - p.r, p.x))
        p.y = max(p.r, min(H - p.r, p.y))

    def _spawn_bullet(self):
        # spawn on border
        side = random.randint(0, 3)
        if side == 0:      # top
            x, y = random.uniform(0, W), 0.0
        elif side == 1:    # bottom
            x, y = random.uniform(0, W), float(H)
        elif side == 2:    # left
            x, y = 0.0, random.uniform(0, H)
        else:              # right
            x, y = float(W), random.uniform(0, H)

        # fixed target point inside map (NOT player position)
        tx = random.uniform(0, W)
        ty = random.uniform(0, H)

        dx = tx - x
        dy = ty - y
        norm = math.hypot(dx, dy)
        if norm < 1e-9:
            dx, dy, norm = 1.0, 0.0, 1.0

        vx = self.bullet_speed * (dx / norm)
        vy = self.bullet_speed * (dy / norm)

        self.bullets.append(Bullet(x, y, vx, vy))

        if len(self.bullets) > self.max_bullets:
            self.bullets = self.bullets[-self.max_bullets:]

    @staticmethod
    def _circle_hit(ax, ay, ar, bx, by, br) -> bool:
        dx = ax - bx
        dy = ay - by
        rr = ar + br
        return (dx * dx + dy * dy) <= (rr * rr)

    def step(self, action: int, dt: float):
        """
        action: 0 up, 1 down, 2 left, 3 right, 4 stay
        dt: seconds
        returns: reward, done
        """
        # If episode already done, do nothing (play loop can still reset anytime)
        if self.done:
            return 0.0, True

        p = self.player

        # move player
        if action == 0:
            p.y -= p.speed * dt
        elif action == 1:
            p.y += p.speed * dt
        elif action == 2:
            p.x -= p.speed * dt
        elif action == 3:
            p.x += p.speed * dt
        elif action == 4:
            pass

        self._clamp_player()

        # spawn bullets at fixed interval
        self._spawn_acc += dt
        while self._spawn_acc >= self.spawn_interval:
            self._spawn_acc -= self.spawn_interval
            self._spawn_bullet()

        # move bullets
        for b in self.bullets:
            b.x += b.vx * dt
            b.y += b.vy * dt

        # remove off-screen bullets (with margin)
        margin = 60.0
        self.bullets = [
            b for b in self.bullets
            if (-margin <= b.x <= W + margin) and (-margin <= b.y <= H + margin)
        ]

        # collision
        if any(self._circle_hit(p.x, p.y, p.r, b.x, b.y, b.r) for b in self.bullets):
            self.done = True
            return -200.0, True

        # time / win condition
        self.t += dt
        if self.t >= self.survival_seconds:
            self.done = True
            return 200.0, True

        # shaping
        reward = 1.0
        if action == 4:
            reward -= self.idle_penalty
        return reward, False
