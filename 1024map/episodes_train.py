import numpy as np
import matplotlib.pyplot as plt
import pygame

from core_env import BulletHellEnv, W, H

# ============================================================
# ====================== TRAINING PARAMS =====================
# ============================================================
TOT_EPISODES = 15000
SHOW_EVERY = 1000

LR = 0.1
DISCOUNT = 0.95

epsilon = 1.0
EPS_DECAY = 0.9997
EPS_MIN = 0.05

ACTIONS = 5  # up, down, left, right, stay

DT = 1.0 / 60.0          # fixed simulation timestep
MAX_STEPS = 12000        # safety cap


# ============================================================
# ====================== STATE DESIGN ========================
# ============================================================
SECTORS = 4
TIME_BINS = 10

DANGER_NEAR = 45.0
DANGER_MID  = 140.0
WALL_MARGIN = 40.0


def sector_index(dx, dy):
    # Right(0), Up(1), Left(2), Down(3)
    if abs(dx) >= abs(dy):
        return 0 if dx > 0 else 2
    else:
        return 3 if dy > 0 else 1


def get_state(env: BulletHellEnv):
    p = env.player
    min_dist = [1e9] * SECTORS

    for b in env.bullets:
        dx = b.x - p.x
        dy = b.y - p.y
        s = sector_index(dx, dy)
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < min_dist[s]:
            min_dist[s] = dist

    danger = []
    for d in min_dist:
        if d <= DANGER_NEAR:
            danger.append(2)
        elif d <= DANGER_MID:
            danger.append(1)
        else:
            danger.append(0)

    near_wall = 1 if (
        p.x <= WALL_MARGIN or p.x >= W - WALL_MARGIN or
        p.y <= WALL_MARGIN or p.y >= H - WALL_MARGIN
    ) else 0

    frac = min(1.0, env.t / env.survival_seconds)
    time_bin = min(TIME_BINS - 1, int(frac * TIME_BINS))

    return tuple(danger + [near_wall, time_bin])


def all_states():
    for a in [0, 1, 2]:
        for b in [0, 1, 2]:
            for c in [0, 1, 2]:
                for d in [0, 1, 2]:
                    for nw in [0, 1]:
                        for tb in range(TIME_BINS):
                            yield (a, b, c, d, nw, tb)


# ============================================================
# ====================== Q TABLE =============================
# ============================================================
q_table = {}
for s in all_states():
    q_table[s] = [np.random.uniform(-1.0, 0.0) for _ in range(ACTIONS)]


# ============================================================
# ====================== ENV INIT ============================
# ============================================================
env = BulletHellEnv(
    spawn_interval=0.12,
    bullet_speed=220.0,
    survival_seconds=40.0,
    idle_penalty=0.9,
)


# ============================================================
# ====================== VISUALIZATION =======================
# ============================================================
def render_episode(env: BulletHellEnv, q_table):
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Training Visualization")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    env.reset()

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        obs = get_state(env)
        action = int(np.argmax(q_table[obs]))

        reward, done = env.step(action, dt)

        screen.fill((18, 18, 22))

        for b in env.bullets:
            pygame.draw.circle(screen, (210, 210, 210),
                               (int(b.x), int(b.y)), int(b.r))

        p = env.player
        pygame.draw.circle(screen, (255, 80, 80),
                           (int(p.x), int(p.y)), int(p.r))

        hud = font.render(
            f"t={env.t:.2f}s bullets={len(env.bullets)} reward={reward:.1f}",
            True, (240, 240, 240)
        )
        screen.blit(hud, (12, 12))

        pygame.display.flip()

        if done:
            pygame.time.delay(800)
            break

    pygame.quit()


# ============================================================
# ====================== TRAIN LOOP ==========================
# ============================================================
episode_rewards = []

for episode in range(TOT_EPISODES):
    env.reset()
    episode_reward = 0.0

    if episode % SHOW_EVERY == 0:
        print(f"On episode number {episode}, epsilon value is {epsilon}")
        if len(episode_rewards) >= SHOW_EVERY:
            print(f"Mean for last {SHOW_EVERY} episodes : "
                  f"{np.mean(episode_rewards[-SHOW_EVERY:])}")

        # ---- visualize current policy (no training) ----
        render_episode(env, q_table)

    for _ in range(MAX_STEPS):
        obs = get_state(env)

        if np.random.random() > epsilon:
            action = int(np.argmax(q_table[obs]))
        else:
            action = np.random.randint(0, ACTIONS)

        reward, done = env.step(action, DT)
        episode_reward += reward

        new_obs = get_state(env)

        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if done:
            new_q = reward
        else:
            new_q = (1 - LR) * current_q + LR * (reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q

        if done:
            break

    episode_rewards.append(episode_reward)
    epsilon = max(EPS_MIN, epsilon * EPS_DECAY)


# ============================================================
# ====================== PLOT ================================
# ============================================================
moving_avg = np.convolve(
    episode_rewards,
    np.ones((SHOW_EVERY,)) / SHOW_EVERY,
    mode="valid"
)

plt.plot(range(len(moving_avg)), moving_avg)
plt.xlabel("Episode Number")
plt.ylabel(f"Average reward per {SHOW_EVERY} episodes")
plt.show()
