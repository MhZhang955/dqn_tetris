import os

import torch
import numpy as np
import random
from tetris import Tetris
from dqn_agent_with_batch_norm import DQNAgent
from tqdm import tqdm


def set_seed(seed=42):
    """Fix all random seeds"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
    # Hyperparameters
    config = {
        "episodes": 600000,
        "batch_size": 128,
        "n_neurons": [128, 128, 128],
        "lr": 1e-3,
        "epsilon_stop_episode": 590000,
        "mem_size": 20000,
        "replay_start_size": 5000,
        "grad_clip": 0.5,
        "save_every": 100,
        "log_every": 10
    }

    # dir for saving model by root dir and epoch
    model_dir = 'models_result/dqn_norm/' + str(config["episodes"]) + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Initialize
    set_seed()
    env = Tetris(tick_rate=10000)
    agent = DQNAgent(
        state_size=env.get_state_size(),
        n_neurons=config["n_neurons"],
        lr=config["lr"],
        epsilon_stop_episode=config["epsilon_stop_episode"],
        mem_size=config["mem_size"],
        replay_start_size=config["replay_start_size"],
        grad_clip=config["grad_clip"]
    )

    # Training loop
    best_score = -float('inf')
    scores = []

    for episode in tqdm(range(config["episodes"]), desc="Training"):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Get action
            next_states = env.get_next_states()
            if not next_states:  # No valid moves
                done = True
                break

            states = [torch.FloatTensor(s) for s in next_states.values()]
            best_idx = agent.best_state(states)
            best_action = list(next_states.keys())[best_idx]

            # Execute action
            reward, done = env.play(best_action[0], best_action[1], render=False)
            next_state = next_states[best_action]
            total_reward += reward

            # Store transition
            agent.add_to_memory(state, next_state, reward, done)
            state = next_state

        # Train after episode
        if len(agent.memory) >= config["replay_start_size"]:
            loss = agent.train(config["batch_size"])

            # Logging
            if episode % config["log_every"] == 0:
                agent.writer.add_scalar("Episode/score", total_reward, episode)
                agent.writer.add_scalar("Episode/length", env.get_game_score(), episode)

        scores.append(total_reward)

        # Save best model
        if total_reward > best_score:
            best_score = total_reward
            agent.save_model(model_dir + "best_dqn_norm.pth")
            tqdm.write(f"🔥 New best at episode {episode}: Score={best_score}")

        # Periodic save
        if episode % config["save_every"] == 0:
            agent.save_model(f"{model_dir}checkpoint_ep{episode}.pth")

    # Final report
    agent.writer.close()
    print(f"\n🏆 Training completed! Best score: {best_score}")
    print(f"Average last 100 scores: {np.mean(scores[-100:]):.1f}")


if __name__ == "__main__":
    train()