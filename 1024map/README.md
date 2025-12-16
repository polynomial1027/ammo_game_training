
# Ammo Game Training (Bullet Hell RL)

This repository implements a **bullet-hell (danmaku) avoidance game** in a **1024×1024 continuous 2D world** and trains an agent using **tabular Q-learning**.

- The agent controls a player that moves **up/down/left/right/stay**
- Bullets spawn from the **screen borders** and travel in **fixed straight-line trajectories**
- Training runs headless for speed, and **visualizes one full episode every N episodes** using pygame

Reference/inspiration: https://github.com/atul-g/dodge_the_ammo

---

## Files

- `core_env.py`  
  Continuous environment logic (NO pygame dependency).  
  Defines player dynamics, bullet spawning, bullet motion, collision, termination, and **reward function**.

- `episodes_train.py`  
  Q-learning training loop + periodic pygame visualization.

- `play_pygame.py`  
  Manual play / visual sanity check.

---

## Reward Mechanism (Current)

The reward is produced inside:

**`core_env.py` → `BulletHellEnv.step(action, dt)`**

### Terminal rewards
- **Hit by a bullet**: `reward = -200.0` and `done = True`
- **Survive until time limit** (`survival_seconds`): `reward = +200.0` and `done = True`

### Shaping reward (per step while alive)
- **Base survival reward per step**: `reward = +1.0`
- **Idle penalty** (if action is `stay`): `reward -= idle_penalty`  
  (default `idle_penalty = 0.2`)

This encourages survival while discouraging trivial idle behavior.

---

## Where to Adjust Rewards

### Shaping rewards
File: **`core_env.py`**

```python
reward = 1.0
if action == 4:
    reward -= self.idle_penalty
```

Adjust:
- `reward = 1.0` → per-step survival incentive
- `idle_penalty` → how strongly the agent is forced to move

You can also change the default via the constructor:
```python
BulletHellEnv(..., idle_penalty=0.2)
```

### Terminal rewards
File: **`core_env.py`**

```python
return -200.0, True   # hit
return  200.0, True   # survive
```

---

## Adjusting Difficulty

All parameters below are passed when creating `BulletHellEnv` in `episodes_train.py`.

- **Bullet frequency**: `spawn_interval` (seconds per bullet)
- **Bullet speed**: `bullet_speed` (px/s)
- **Episode length**: `survival_seconds`

Player and bullet hitboxes are defined in `core_env.py`:
- `Player.r`
- `Bullet.r`
- `Player.speed`

---

## Training Configuration

File: **`episodes_train.py`**

- `TOT_EPISODES` – total training episodes
- `SHOW_EVERY` – print stats and visualize every N episodes
- `LR`, `DISCOUNT` – Q-learning parameters
- `epsilon`, `EPS_DECAY`, `EPS_MIN` – exploration schedule
- `DT` – simulation timestep
- State discretization:
  - `DANGER_NEAR`, `DANGER_MID`
  - `WALL_MARGIN`
  - `TIME_BINS`

---

## How to Run

Install dependencies:
```bash
pip install numpy matplotlib pygame
```

Train with periodic visualization:
```bash
python episodes_train.py
```

---

## Practical Tuning Tips

- If learning is unstable: reduce bullet speed or spawn rate.
- If agent idles too much: increase `idle_penalty`.
- If agent hugs walls: adjust `WALL_MARGIN` or remove the feature.
- For harder setups, consider curriculum learning.

---

## Roadmap

- Curriculum learning (gradual difficulty increase)
- Richer observations (density / direction features)
- PPO / DQN agents
- Recording gameplay videos for documentation

---
