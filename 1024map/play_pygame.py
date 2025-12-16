# play_pygame.py
import pygame
from core_env import BulletHellEnv, W, H

def action_from_keys(keys) -> int:
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        return 0
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        return 1
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        return 2
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        return 3
    return 4

def main():
    pygame.init()
    pygame.display.set_caption("Ammo Game Training - Bullet Hell (pygame)")

    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 20)

    env = BulletHellEnv(
        spawn_interval=0.12,
        bullet_speed=220.0,
        survival_seconds=20.0,
        idle_penalty=0.2,
    )
    env.reset()

    # auto-reset timer (seconds after death/win)
    AUTO_RESET_DELAY = 1.5
    done_elapsed = 0.0

    running = True
    while running:
        dt = clock.tick(120) / 1000.0

        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()

        # step env only if not done
        if not env.done:
            action = action_from_keys(keys)
            reward, _ = env.step(action, dt)
        else:
            reward = 0.0
            done_elapsed += dt
            if done_elapsed >= AUTO_RESET_DELAY:
                env.reset()
                done_elapsed = 0.0

        # render
        screen.fill((18, 18, 22))

        for b in env.bullets:
            pygame.draw.circle(screen, (210, 210, 210), (int(b.x), int(b.y)), int(b.r))

        p = env.player
        pygame.draw.circle(screen, (255, 80, 80), (int(p.x), int(p.y)), int(p.r))

        line1 = f"t={env.t:5.2f}s  bullets={len(env.bullets):4d}  reward={reward:6.1f}  done={env.done}"
        line2 = "Move: WASD/Arrows | Quit: ESC | Auto-reset after done"
        hud1 = font.render(line1, True, (240, 240, 240))
        hud2 = font.render(line2, True, (240, 240, 240))
        screen.blit(hud1, (12, 12))
        screen.blit(hud2, (12, 36))

        if env.done:
            msg = f"DONE - restarting in {max(0.0, AUTO_RESET_DELAY - done_elapsed):.1f}s"
            hud3 = font.render(msg, True, (255, 230, 120))
            screen.blit(hud3, (12, 60))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()

