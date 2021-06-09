import numpy as np
from reinforce_tf2 import Agent
from libraries import EnvironmentGateIO
from datetime import datetime

if __name__ == '__main__':
    agent = Agent(alpha=0.0005, gamma=0.99, n_actions=3)

    env = EnvironmentGateIO(money_USDT=5, money_CRYP=0.007)
    score_history = []
    print("Environment loaded")
    num_episodes = 2000

    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        print(i, "- iteration start")
        timer = 0
        one_time = False
        while not done:
            start_time = datetime.now()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward)
            observation = observation_
            score += reward
            if one_time == False:
                one_time = True
                print("Time to one interation = ", datetime.now() - start_time)
                print("Time to full work = ", (datetime.now() - start_time)*60*60*2)
            if info > timer+5:
                print("=", end="")
                timer = info
        print("> done")
        score_history.append(score)

        agent.learn()
        avg_score = np.mean(score_history[-100:])
        print('episode: ', i, 'score: %.1f' % score,
              'average score %.1f' % avg_score)
        agent.policy.save_weights('gradient_policy-' + str(i) + '.ckpt')

    filename = 'lunar-lander.png'
