"""
第9次作业, 实现表格型Q-Learning算法
"""
import random
import numpy as np
from office_world import Game
from rl_test import run_test


class LearningParams:
    def __init__(self, epsilon=0.1, lr=0.1, gamma=0.9):
        """
        :param epsilon: epsilon-greedy的随机探索权重
        :param lr: 学习率
        :param gamma: 折扣因子
        """
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma


class TestingParams:
    def __init__(self, test_freq=1, max_ep_len=100):
        """
        :param test_freq: 测试频率, 即每测试一次要训练多少步
        :param max_ep_len: 单个episode的最大长度
        """
        self.test_freq = test_freq
        self.max_ep_len = max_ep_len


class QLAgent:
    def __init__(self, num_states, num_actions, learning_params):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_params = learning_params

    def learn(self, s1, a, r, s2, done):
        pass

    def get_action(self, state, eval_mode=False):
        pass


def run_experiment(env, agent, testing_params, num_episode=1000):
    for i_ep in range(num_episode):
        s1 = env.reset()
        for t in range(testing_params.max_ep_len):
            a = agent.get_action(s1)
            s2, reward, done, _ = env.step(a)
            # learn()中自行添加参数
            agent.learn()
            # 转移到下一步
            s1 = s2
            # 若环境已终止, 则进入下一个episode的训练
            if done:
                break
        # 每训练test_freq个episode执行一次测试
        if i_ep % testing_params.test_freq == 0:
            # 测试是在env的副本上执行, 不改变原来的env信息
            test_reward = run_test(env.task, testing_params.max_ep_len, agent)
            print("Training episodes: %d, reward: %.2f" % (i_ep, test_reward))



if __name__ == "__main__":
    learning_params = LearningParams()  # 定义学习超参数
    testing_params = TestingParams()  # 定义测试超参数
    task = 'get_coffee'  # 定义任务
    env = Game(task)  # 初始化环境
    # 获取环境的状态,动作数
    n_states, n_actions = env.num_features, env.num_actions
    # 定义Q-Learning智能体
    agent = QLAgent(num_states=n_states, num_actions=n_actions, learning_params=learning_params)
    # 定义随机种子, 保证实验结果可复现
    random.seed(0)
    # 运行实验
    run_experiment(env, agent, testing_params)
