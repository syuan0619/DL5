"""
DQN Lab 5 完整實作 (修正版)
更新日期：2025-05-22
主要修正：
1. 解決參數重複定義問題
2. 統一參數命名規範
3. 增強異常處理與資源管理
4. 優化評估錄影功能
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import os
from collections import deque
import argparse
import wandb
import imageio
from typing import Optional, List, Tuple

class AtariPreprocessor:
    """Atari環境預處理器（灰度化+縮放+幀堆疊）"""
    def __init__(self, frame_stack: int = 4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs: np.ndarray) -> np.ndarray:
        """單幀處理流程"""
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs: np.ndarray) -> np.ndarray:
        """環境重置時初始化幀堆疊"""
        frame = self.preprocess(obs)
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs: np.ndarray) -> np.ndarray:
        """每一步更新幀堆疊"""
        self.frames.append(self.preprocess(obs))
        return np.stack(self.frames, axis=0)

class PrioritizedReplayBuffer:
    """優先經驗回放（PER）實現"""
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.max_priority = 1.0

    def add(self, transition: Tuple) -> None:
        """添加經驗並初始化優先級"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = self.max_priority ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """根據優先級採樣"""
        if len(self.buffer) == 0:
            return [], [], []

        probs = self.priorities[:len(self.buffer)] / self.priorities[:len(self.buffer)].sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices: List[int], errors: np.ndarray) -> None:
        """更新採樣經驗的優先級"""
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + 1e-5) ** self.alpha
            self.max_priority = max(self.max_priority, self.priorities[idx])

class DQNNetwork(nn.Module):
    """支援多種輸入的DQN網路架構"""
    def __init__(self, input_shape: Tuple, num_actions: int):
        super().__init__()
        if len(input_shape) == 1:  # CartPole等低維狀態
            self.net = nn.Sequential(
                nn.Linear(input_shape[0], 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions)
            )
        else:  # Atari等高維圖像
            self.net = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64*7*7, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        if len(x.shape) == 3:  # 圖像歸一化
            x = x / 255.0
        return self.net(x)

class DQNAgent:
    """整合強化學習三大增強技術的智能體"""
    def __init__(self, env_name: str, args: argparse.Namespace):
        # 環境初始化
        self.env = gym.make(env_name)
        self.test_env = gym.make(env_name, render_mode="rgb_array", width=1280, height=720) if args.record_video else None
        self.num_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 參數設定（安全獲取）
        self.gamma = getattr(args, 'gamma', 0.99)
        self.epsilon = getattr(args, 'epsilon_start', 1.0)
        self.epsilon_decay = getattr(args, 'epsilon_decay', 0.995)
        self.epsilon_min = getattr(args, 'epsilon_min', 0.01)
        self.batch_size = args.batch_size
        self.n_steps = args.n_steps
        self.target_update_freq = args.target_update_freq

        # 狀態預處理
        self.state_shape = (self.env.observation_space.shape[0],) if len(self.env.observation_space.shape) == 1 else (4, 84, 84)
        self.preprocessor = None if len(self.env.observation_space.shape) == 1 else AtariPreprocessor(frame_stack=4)

        # 神經網路
        self.q_net = DQNNetwork(self.state_shape, self.num_actions).to(self.device)
        self.target_net = DQNNetwork(self.state_shape, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        # 經驗回放
        self.memory = PrioritizedReplayBuffer(args.memory_size) if args.use_per else deque(maxlen=args.memory_size)
        self.train_steps = 0

        # 模型保存
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state: np.ndarray, epsilon: Optional[float] = None) -> int:
        """ε-greedy策略選擇動作"""
        epsilon = epsilon if epsilon is not None else self.epsilon
        if random.random() < epsilon:
            return self.env.action_space.sample()
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def train(self) -> float:
        """執行單次訓練步驟"""
        if len(self.memory) < self.batch_size:
            return 0.0

        # 採樣批次
        if isinstance(self.memory, PrioritizedReplayBuffer):
            batch, indices, weights = self.memory.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch = random.sample(self.memory, self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)

        # 準備數據
        states = torch.FloatTensor(np.array([x[0] for x in batch])).to(self.device)
        actions = torch.LongTensor([x[1] for x in batch]).to(self.device)
        rewards = torch.FloatTensor([x[2] for x in batch]).to(self.device)
        next_states = torch.FloatTensor(np.array([x[3] for x in batch])).to(self.device)
        dones = torch.FloatTensor([x[4] for x in batch]).to(self.device)

        # 計算目標值
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            next_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            targets = rewards + (1 - dones) * self.gamma * next_values

        # 計算損失
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        td_errors = current_q - targets

        # 更新優先級
        if isinstance(self.memory, PrioritizedReplayBuffer):
            self.memory.update_priorities(indices, td_errors.cpu().detach().numpy())

        # 優化
        loss = (weights * td_errors.pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # 更新目標網路
        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def run_episode(self, train_mode: bool = True, render: bool = False) -> Tuple[float, Optional[List[np.ndarray]]]:
        """執行完整回合"""
        state, _ = self.env.reset()
        if self.preprocessor:
            state = self.preprocessor.reset(state)

        episode_reward = 0
        n_step_buffer = deque(maxlen=self.n_steps)
        frames = [] if render else None

        while True:
            if render and self.test_env:
                frames.append(self.test_env.render())

            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            if self.preprocessor:
                next_state = self.preprocessor.step(next_state)

            # 儲存n-step轉移
            n_step_buffer.append((state, action, reward, next_state, done))
            if len(n_step_buffer) == self.n_steps:
                states, acts, rews, next_states, dones = zip(*n_step_buffer)
                transition = (states[0], acts[0], sum(rews), next_states[-1], dones[-1])
                if isinstance(self.memory, PrioritizedReplayBuffer):
                    self.memory.add(transition)
                else:
                    self.memory.append(transition)

            state = next_state
            episode_reward += reward

            if train_mode:
                loss = self.train()
                if loss > 0:
                    wandb.log({
                        "train/loss": loss,
                        "train/epsilon": self.epsilon,
                        "train/reward": reward
                    })

            if done:
                break

        # 衰減探索率
        if train_mode and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return episode_reward, frames

    def evaluate(self, num_episodes: int = 3) -> None:
        """評估模型表現"""
        if not self.test_env:
            raise ValueError("測試環境未初始化，請設置 --record_video")

        for ep in range(num_episodes):
            _, frames = self.run_episode(train_mode=False, render=True)
            if frames:
                video_path = os.path.join(self.save_dir, f"eval_ep{ep}.mp4")
                imageio.mimsave(video_path, frames, fps=30)
                print(f"評估影片已保存至 {video_path}")

    def save_model(self, filename: str) -> None:
        """保存模型參數"""
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        print(f"模型已保存至 {path}")

    def load_model(self, filename: str) -> None:
        """加載模型參數"""
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"已從 {path} 加載模型")

def get_args() -> argparse.Namespace:
    """定義並解析命令行參數"""
    parser = argparse.ArgumentParser(description="DQN Lab 5 完整實作")
    
    # 環境參數組
    env_group = parser.add_argument_group('Environment')
    env_group.add_argument("--env", default="CartPole-v1",
                          choices=["CartPole-v1", "ALE/Pong-v5"],
                          help="訓練環境名稱")
    env_group.add_argument("--record_video", action="store_true",
                          help="是否錄製評估影片")

    # 訓練參數組
    train_group = parser.add_argument_group('Training')
    train_group.add_argument("--episodes", type=int, default=1000,
                            help="訓練總回合數")
    train_group.add_argument("--lr", type=float, default=1e-4,
                            help="學習率")
    train_group.add_argument("--batch_size", type=int, default=32,
                            help="批次大小")
    train_group.add_argument("--memory_size", type=int, default=100000,
                            help="經驗回放緩衝區大小")
    train_group.add_argument("--target_update_freq", type=int, default=1000,
                            help="目標網路更新頻率")

    # RL參數組
    rl_group = parser.add_argument_group('Reinforcement Learning')
    rl_group.add_argument("--gamma", type=float, default=0.99,
                          help="折扣因子 (範圍: 0.9~0.999)")
    rl_group.add_argument("--epsilon_start", type=float, default=1.0,
                          help="初始探索率")
    rl_group.add_argument("--epsilon_decay", type=float, default=0.995,
                          help="探索率衰減係數")
    rl_group.add_argument("--epsilon_min", type=float, default=0.01,
                          help="最小探索率")

    # 增強技術組
    tech_group = parser.add_argument_group('Advanced Techniques')
    tech_group.add_argument("--use_per", action="store_true",
                           help="啟用優先經驗回放")
    tech_group.add_argument("--n_steps", type=int, default=3,
                           help="Multi-Step Return步數")

    # 系統設定組
    sys_group = parser.add_argument_group('System')
    sys_group.add_argument("--save_dir", default="./saved_models",
                          help="模型保存目錄")
    sys_group.add_argument("--seed", type=int, default=42,
                          help="隨機種子")

    return parser.parse_args()

def validate_args(args: argparse.Namespace) -> None:
    """參數有效性驗證"""
    if not 0 < args.gamma <= 1:
        raise ValueError("gamma 必須在 (0,1] 範圍內")
    if not 0 <= args.epsilon_start <= 1:
        raise ValueError("epsilon_start 必須在 [0,1] 範圍內")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

def main():
    args = get_args()
    validate_args(args)

    # 設定隨機種子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 初始化W&B
    wandb.init(project="DLP-Lab5", config=vars(args))

    try:
        # 訓練流程
        agent = DQNAgent(args.env, args)
        best_reward = -np.inf

        for ep in range(args.episodes):
            episode_reward, _ = agent.run_episode(train_mode=True)

            # 定期評估
            if ep % 100 == 0 and args.record_video:
                agent.evaluate(num_episodes=1)
                agent.save_model(f"{args.env}_ep{ep}.pt")

            # 保存最佳模型
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save_model(f"{args.env}_best.pt")

            wandb.log({
                "episode_reward": episode_reward,
                "epsilon": agent.epsilon
            })
            print(f"Episode {ep:4d} | Reward: {episode_reward:6.1f} | Epsilon: {agent.epsilon:.3f}")

        # 最終保存
        agent.save_model(f"{args.env}_final.pt")

    finally:
        # 確保資源釋放
        if 'agent' in locals():
            if hasattr(agent, 'env'):
                agent.env.close()
            if hasattr(agent, 'test_env') and agent.test_env:
                agent.test_env.close()
        wandb.finish()

if __name__ == "__main__":
    main()
