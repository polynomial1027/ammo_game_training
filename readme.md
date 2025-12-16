# Ammo Game Training

This project explores **reinforcement learning for a simplified bullet-hell (danmaku) style avoidance game**.

An agent controls a small ship on a 2D grid and learns to **survive as long as possible** by dodging bullets that are fired from the edges of the screen along fixed straight-line trajectories.

The project focuses on **environment design, state abstraction, and reward shaping**, rather than deep neural networks, and currently uses **tabular Q-learning**.

---

## Project Overview

- **Game type**: Simplified bullet-hell / obstacle-avoidance game  
- **Agent control**: Discrete actions on a 2D grid  
- **Bullets**:
  - Spawn randomly from screen boundaries  
  - Travel along **fixed straight lines** (no homing / tracking)  
  - Move slower than the agent to allow learnable avoidance strategies  
- **Objective**: Maximize survival time without being hit  

The training process demonstrates how learning behavior emerges as the agent gradually discovers safer movement patterns under stochastic bullet patterns.

---

## Reinforcement Learning Setup

- **Algorithm**: Tabular Q-learning  
- **State representation** (discrete):
  - Local danger levels in 4 directional sectors around the agent  
  - Whether the agent is near the boundary  
  - A coarse time bin representing episode progress  
- **Action space**:
  - Move up / down / left / right (faster than bullets)
  - Stay (penalized to avoid trivial strategies)
- **Reward design**:
  - Positive reward for each survival step  
  - Penalty for staying idle  
  - Large negative reward when hit  
  - Bonus reward for surviving a full episode  

The environment difficulty is **fixed during training** to ensure convergence before experimenting with curriculum learning.

---

## Visualization

The game state is rendered periodically during training using a simple grid-based visualization, allowing direct inspection of the agent’s learned behavior.

---

## Motivation

This project is intended as:

- A **learning-oriented reinforcement learning environment**
- A clean example of how **environment design affects learnability**
- A stepping stone toward more advanced methods (e.g. PPO / DQN) and richer state representations

---

## Reference

This project is **inspired by and adapted from** the following repository:

- https://github.com/atul-g/dodge_the_ammo

The original project demonstrates a 1D bullet-dodging environment solved via Q-learning.  
This repository extends the idea to a **2D bullet-hell setting with straight-line projectiles**, emphasizing environment dynamics and training stability.

---

## Notes

- This is an experimental and educational project.
- The code is intentionally kept simple and readable.
- Future extensions may include curriculum learning, richer state features, or neural-network-based agents.

