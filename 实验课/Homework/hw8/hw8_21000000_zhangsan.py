import numpy as np
from grid_env import MiniWorld


class DPAgent:
    def __init__(self, env):
        self.env = env
        self.n_state = env.observation_space.n
        self.n_action = env.action_space.n

    def iteration(self, threshold=1e-3):
        values = np.zeros([self.n_state])
        # 我们约定a=0,1,2,3 对应动作["^", ">", "v", "<"]
        # policy[s][a]为在状态s下采取动作a的概率
        policy = np.zeros([self.n_state, self.n_action])
        """
        在这里编写价值迭代算法或策略迭代算法
        """
        return values, policy


if __name__ == "__main__":
    env = MiniWorld()
    agent = DPAgent(env)

    values, policy = agent.iteration(threshold=1e-3)

    env.show_values(values, sec=3)  # 将values渲染成颜色格子, 显示3秒
    env.show_policy(policy)  # 在终端打印每个状态下的动作
