import gym

def run_test(agent):
    r_list = []
    test_time = 5
    for seed in range(test_time):
        env = gym.make("CartPole-v1", render_mode="human")
        s1 = env.reset(seed=seed)[0]
        # Starting interaction with the environment
        r_total = 0
        for t in range(200):
            a = agent.get_action(s1, eval_mode=True)
            s2, reward, done, _, _ = env.step(a)
            r_total += reward
            # Restarting the environment (Game Over)
            if done:
                break
            # Moving to the next state
            s1 = s2
        r_list.append(r_total)
    return sum(r_list)/test_time


if __name__ == "__main__":
    # 将hw10_21000000_zhangsan改成自己的信息来测试
    file_name = "hw10_21000000_zhangsan"
    exec("from %s import MyAgent"%file_name)
    agent = MyAgent()
    agent.load_model(file_name)
    r_total = run_test(agent)
    print("total reward: %.2f" % r_total)
