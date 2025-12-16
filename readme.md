# Ammo Game Training

A small reinforcement-learning (RL) project that explores **training an agent to survive a bullet-hell / obstacle-avoidance game**.

This repository contains **two implementations** of the same idea:

- `original_version/` — the earlier **grid / pixel-style** environment (tabular Q-learning on a small discrete map).
- `1024map/` — the newer **1024×1024 continuous** environment rendered with **pygame**, with bullets fired in straight lines from screen borders.

The overall objective in both versions is:

> **Survive as long as possible** while avoiding bullets.  
> Surviving longer yields higher return; getting hit ends the episode with a large penalty.

Reference / inspiration: https://github.com/atul-g/dodge_the_ammo

---

## Repository Layout

```
ammo_game_training/
├─ README.md
├─ requirements.txt
├─ original_version/
│  ├─ env.py
│  ├─ episodes.py
│  └─ (optional assets / notes)
└─ 1024map/
   ├─ core_env.py
   ├─ play_pygame.py
   └─ episodes_train.py
```

> Filenames may differ slightly depending on your local edits, but the intent is:
> - `env.py` / `core_env.py` define the environment and dynamics
> - `episodes.py` / `episodes_train.py` run training episodes
> - `play_pygame.py` is for manual play / visualization sanity checks

---

## Version 1: `original_version/` (Pixel / Grid)

### What it is
A compact **grid-world** bullet dodging environment:

- Small discrete grid (e.g., 6×10 or 15×15 depending on your snapshot)
- Discrete actions (left/right/stay or 4-direction depending on the script)
- Bullets advance in discrete steps
- Trained with **tabular Q-learning** using a small discrete state representation (e.g., positions and/or danger sectors)

### How to run training
From the repo root:

```bash
cd original_version
python episodes.py
```

Typical behavior:
- Runs `TOT_EPISODES` episodes
- Prints progress every `SHOW_EVERY` episodes
- Periodically renders an episode to visualize the current policy (OpenCV/PIL-style rendering)

---

## Version 2: `1024map/` (1024×1024 pygame, Continuous)

### What it is
A more game-like bullet-hell environment:

- World size: **1024×1024**
- Player moves in continuous coordinates with discrete actions:
  - `0=up, 1=down, 2=left, 3=right, 4=stay`
- Bullets:
  - Spawn from **screen borders**
  - Travel along **fixed straight-line trajectories** (no homing)
  - Spawn frequency and speed are fixed via environment parameters
- Rendering:
  - Uses **pygame** for smooth visuals (not pixel-grid style)

### Core files
- `core_env.py`  
  Defines `BulletHellEnv` (player, bullets, spawning, physics update, collision, termination, reward).
- `episodes_train.py`  
  Trains a Q-table policy with epsilon-greedy exploration, and **visualizes one full episode every `SHOW_EVERY` episodes**.
- `play_pygame.py`  
  Manual play / visual test harness.

---

## Reward Mechanism (1024map)

Rewards are produced in:

**`1024map/core_env.py` → `BulletHellEnv.step(action, dt)`**

Current structure (typical defaults):

- **Per-step survival reward**: `+1.0` while alive
- **Idle penalty**: if action is `stay`, reward is reduced by `idle_penalty` (default `0.2`)
- **Hit by bullet**: terminal reward `-200.0`, `done=True`
- **Survive to time limit**: terminal reward `+200.0`, `done=True`

> Important: An episode ends either by **getting hit** or by **surviving long enough** (time limit).

### Where to adjust rewards
Edit **`1024map/core_env.py`**:
- Shaping:
  - `reward = 1.0`
  - `reward -= self.idle_penalty` when staying still
- Terminal:
  - `return -200.0, True` on hit
  - `return 200.0, True` on survive

---

## Installation

### Recommended (conda)
Create and activate an environment (Python 3.10 recommended):

```bash
conda create -n ammo_rl python=3.10 -y
conda activate ammo_rl
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Minimal dependencies
For the 1024map version, you typically need:

- `numpy`
- `matplotlib`
- `pygame`

The original version may additionally use:
- `opencv-python`
- `Pillow`

---

## How to Use

### A) Run the new 1024map game (manual play)
```bash
cd 1024map
python play_pygame.py
```

Controls:
- Move: `WASD` or arrow keys
- Quit: close the window (or `ESC` if implemented)

### B) Train on 1024map with periodic visualization
```bash
cd 1024map
python episodes_train.py
```

You will see logs like:
```
On episode number 0, epsilon value is 1.0
On episode number 1000, epsilon value is ...
Mean for last 1000 episodes : ...
```

Every `SHOW_EVERY` episodes, a pygame window will pop up to render **one full episode** using the current greedy policy.
![Figure 1](figure_1.png)

### C) Train the original grid version
```bash
cd original_version
python episodes.py
```

---

## Configuration / Tuning Guide (1024map)

### Difficulty knobs (in `episodes_train.py` where env is created)
- Bullet frequency: `spawn_interval` (smaller → more bullets)
- Bullet speed: `bullet_speed` (larger → harder)
- Episode length: `survival_seconds` (longer → harder)
- Idle penalty: `idle_penalty` (larger → forces movement)

Example:
```python
env = BulletHellEnv(
    spawn_interval=0.10,
    bullet_speed=260.0,
    survival_seconds=30.0,
    idle_penalty=0.3,
)
```

### Training knobs (in `episodes_train.py`)
- `TOT_EPISODES`
- `SHOW_EVERY`
- `LR`, `DISCOUNT`
- `epsilon`, `EPS_DECAY`, `EPS_MIN`
- `DT` (simulation timestep)

### State discretization knobs (in `episodes_train.py`)
The Q-table does not observe raw pixels; instead it uses a small discrete state such as:
- nearest-bullet danger level in 4 sectors
- near-wall flag
- time bin

Tune:
- `DANGER_NEAR`, `DANGER_MID`
- `WALL_MARGIN`
- `TIME_BINS`

---

## Notes and Next Steps (Optional)

This project intentionally starts with **tabular Q-learning** because it is easy to debug and interpret.

Natural extensions:
- Curriculum learning (increase density/speed gradually)
- Richer observations (bullet density histogram, direction features)
- Neural agents (DQN / PPO) for harder bullet patterns

---

## License

