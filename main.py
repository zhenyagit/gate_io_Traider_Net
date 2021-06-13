import numpy as np
import os
from reinforce_tf2 import Agent
from libraries import EnvironmentGateIO
from datetime import datetime
import tensorflow

if __name__ == '__main__':
    agent = Agent(alpha=0.0005, gamma=0.99, n_actions=3)
    latest = tensorflow.train.latest_checkpoint('./gradient_policy/')
    agent.policy.load_weights(latest)
    time_interval = 60*30
    env = EnvironmentGateIO(money_USDT=5, money_CRYP=0.007, time_interval=time_interval)
    score_history = []
    print("Environment loaded")
    num_episodes = 2000

    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        timer = 0
        one_time = False
        while not done:
            start_time = datetime.now()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward)
            observation = observation_
            score += reward
            os.system("cls")
            print(i, "- iteration start")
            print("Action = ", action)
            print("Reward = ", reward, "\t\t\tScore = ", env.now_score)
            print("Price = ", env.now_price, "\t\t\tStart price = ", env.start_price)
            print("Len orders = ", len(env.orders), "\t\t\tLen succes = ", len(env.success_orders))
            print("USDT = ", env.now_money_USDT, "\t\t\tCRYP = ", env.now_money_CRYP)
            print("Process = ", info)
            print("Time to one interation = ", datetime.now() - start_time)
            print("Time to full work = ", (datetime.now() - start_time)*time_interval/100*(100-info)/3)
        print("done")
        score_history.append(score)

        agent.learn()
        avg_score = np.mean(score_history[-100:])
        print('episode: ', i, 'score: %.1f' % score,
              'average score %.1f' % avg_score)
        agent.policy.save_weights('./gradient_policy/gradient_policy-' + str(i) + '.ckpt')
