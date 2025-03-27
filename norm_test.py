# run_model.py
import sys
import torch
import pygame
from tetris import Tetris
from dqn_agent_with_batch_norm import DQNAgent
import time


def test_model(model_path, render_delay=0.05, num_episodes=5):
    """Test trained model with visualization"""
    # Initialize
    env = Tetris(tick_rate=1)  # Lower tick_rate for better visualization
    agent = DQNAgent(env.get_state_size())
    agent.load_model(model_path)
    agent.set_train_mode(False)  # Ensure eval mode

    pygame.init()
    font = pygame.font.SysFont('Arial', 25)

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        start_time = time.time()

        while not done:
            # Render game
            env.render()
            score_text = font.render(f'Ep {episode} | Score: {total_reward}', True, (255, 255, 255))
            pygame.display.get_surface().blit(score_text, (10, 10))
            pygame.display.flip()

            # Get action
            next_states = env.get_next_states()
            if not next_states:  # No valid moves
                break

            states = [torch.FloatTensor(s) for s in next_states.values()]
            best_idx = agent.best_state(states)
            best_action = list(next_states.keys())[best_idx]

            # Execute action
            reward, done = env.play(
                best_action[0],
                best_action[1],
                render=False  # We're handling rendering separately
            )
            total_reward += reward
            time.sleep(render_delay)  # Control game speed

        # Episode summary
        duration = time.time() - start_time
        print(f"Episode {episode}: Score={total_reward} | Duration={duration:.1f}s")

    pygame.quit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_model.py path_to_model.pth [render_delay] [num_episodes]")
        sys.exit(1)

    # Parse arguments
    model_path = sys.argv[1]
    render_delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    num_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    test_model(model_path, render_delay, num_episodes)