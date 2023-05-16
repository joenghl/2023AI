from office_world import Game

def run_test(task, num_steps, agent):
    env = Game(task)
    s1 = env.reset()
    # Starting interaction with the environment
    r_total = 0
    for t in range(num_steps):
        a = agent.get_action(s1, eval_mode=True)
        s2, reward, done, _ = env.step(a)
        r_total += reward
        # Restarting the environment (Game Over)
        if done:
            break
        # Moving to the next state
        s1 = s2
    return r_total


