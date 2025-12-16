from env import *

episode_rewards = []

for episode in range(TOT_EPISODES):
    player = agent()
    bh = bullet_hell()
    bh.reset()

    if episode % SHOW_EVERY == 0:
        print(f"On episode number {episode}, epsilon value is {epsilon}")
        if len(episode_rewards) >= SHOW_EVERY:
            print(f"Mean for last {SHOW_EVERY} episodes : {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_rew = 0
    done = False

    for step_i in range(SURVIVE_GOAL_STEPS + 1):
        obs = get_state(player, bh, step_i)

        if np.random.random() > epsilon:
            action = int(np.argmax(q_table[obs]))
        else:
            action = np.random.randint(0, ACTIONS)

        player.action(action)
        bh.step()

        if bh.hit(player.x, player.y):
            reward = -HIT_PENALTY
            done = True
        elif step_i >= SURVIVE_GOAL_STEPS:
            reward = SURVIVE_REWARD
            done = True
        else:
            reward = STEP_REWARD
            if action == 4:
                reward -= IDLE_PENALTY

        new_obs = get_state(player, bh, step_i + 1)

        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == SURVIVE_REWARD:
            new_q = SURVIVE_REWARD
        elif reward == -HIT_PENALTY:
            new_q = -HIT_PENALTY
        else:
            new_q = (1 - LR) * current_q + LR * (reward + DISCOUNT * max_future_q)

        q_table[obs][action] = new_q

        if show:
            game = np.zeros((GAME_LENGTH, GAME_WIDTH, 3), dtype=np.uint8)

            for b in bh.bullets:
                game[b[1]][b[0]] = d[BULLET]

            game[player.y][player.x] = d[AGENT]

            img = Image.fromarray(game, 'RGB')
            img = img.resize((420, 420), resample=Image.NEAREST)
            draw = ImageDraw.Draw(img)
            draw.text((6, 6), f"Episode:{episode} Step:{step_i} Eps:{epsilon:.3f}", (255, 255, 255))

            cv2.imshow("bullet_hell_straight", np.array(img))
            if done:
                if cv2.waitKey(600) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

        episode_rew += reward
        if done:
            break

    episode_rewards.append(episode_rew)
    epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Average reward at every {SHOW_EVERY} episodes")
plt.xlabel("Episode Number")
plt.show()
