# dqn_train.py（优化版，添加avg_score日志）
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from dqn_agent import DQNAgent
from tetris import Tetris
from tqdm import tqdm
import time
import numpy as np  # 新增导入


def dqn():
    config = {
        "episodes": 30000,
        "epsilon_stop_episode": 2000,
        "mem_size": 1000,
        "discount": 0.95,
        "batch_size": 128,
        "replay_start_size": 1000,
        "n_neurons": [32, 32, 32],
        "lr": 1e-3,
        "train_epochs": 3,
        "log_every": 50  # 新增：每50个episode记录一次avg_score
    }

    model_dir = 'models_result/dqn/' + str(config["episodes"]) + '/'
    os.makedirs(model_dir, exist_ok=True)

    env = Tetris()
    agent = DQNAgent(
        state_size=env.get_state_size(),
        n_neurons=config["n_neurons"],
        lr=config["lr"],
        epsilon_stop_episode=config["epsilon_stop_episode"],
        mem_size=config["mem_size"],
        discount=config["discount"],
        replay_start_size=config["replay_start_size"]
    )

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"runs/{current_time}_dqn")

    best_score = -float('inf')
    scores = []  # 新增：存储历史得分以计算avg_score
    for episode in tqdm(range(config["episodes"])):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            next_states = env.get_next_states()
            states = [torch.FloatTensor(s) for s in next_states.values()]
            best_idx = agent.best_state(states)
            best_action = list(next_states.keys())[best_idx]

            reward, done = env.play(best_action[0], best_action[1], render=False)
            next_state = next_states[best_action]
            total_reward += reward

            agent.add_to_memory(
                torch.FloatTensor(state),
                torch.FloatTensor(next_state),
                reward,
                done
            )
            state = next_state

        # 训练逻辑保持不变
        if len(agent.memory) >= config["replay_start_size"]:
            for _ in range(config["train_epochs"]):
                loss = agent.train(config["batch_size"])
                writer.add_scalar("Training/loss", loss, episode)

        # 记录当前episode的总reward
        scores.append(total_reward)
        writer.add_scalar("Episode/reward", total_reward, episode)
        writer.add_scalar("Params/epsilon", agent.epsilon, episode)

        # 新增：每隔log_every个episode计算并记录avg_score
        if episode % config["log_every"] == 0 and episode > 0:
            recent_scores = scores[-config["log_every"]:]
            avg_score = np.mean(recent_scores)
            min_score = np.min(recent_scores)
            max_score = np.max(recent_scores)

            writer.add_scalar("Score/avg_score", avg_score, episode)
            writer.add_scalar("Score/min_score", min_score, episode)
            writer.add_scalar("Score/max_score", max_score, episode)
            tqdm.write(f"📊 Episode {episode}: Avg={avg_score:.1f}, Min={min_score}, Max={max_score}")

        # 保存最佳模型逻辑保持不变
        if total_reward > best_score:
            best_score = total_reward
            torch.save(agent.model.state_dict(), model_dir + "best_dqn.pth")
            tqdm.write(f"🔥 New best at episode {episode}: Score={best_score}")

    print(f"\n🏆 Training completed! Best score: {best_score}")
    writer.close()


if __name__ == "__main__":
    dqn()