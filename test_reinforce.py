from reinforce import *
from catch import Catch

from utilities import discount

"""
def discount(rewards, t):
    discounted_reward = 0
    for k in range(t, len(rewards)):
        reward = rewards[k]
        discounted_reward += pow(0.99, k) * reward
    return discounted_reward
"""

"""
0: 244.0
1: 655.0
2: 1091.0
3: 1179.0
4: 1276.0
5: 1360.0
6: 1390.0
7: 1387.0
"""

def main():
    discount_rate = 0.99
    env = Catch(5)
    model = Model(env.observation_space, env.action_space, 0.1)
    observations = []
    rewards = []
    actions = []

    epsiodes = 10000

    for episode in range(0, epsiodes):
        observation = env.reset()

        while True:
            action = model.predict_action(observation)
            observations.append(observation)
            observation, reward, done = env.step(action)
            rewards.append(reward)
            actions.append(action)
            print_score(reward, 10000)

            if done:
                t = 0
                for observation, reward, action in zip(observations, rewards, actions):
                    discounted_reward = pow(discount_rate, t) * discount(rewards, discount_rate, t)
                    model.train(observation, discounted_reward, action)
                    t += 1

                observations = []
                rewards = []
                actions = []
                break

if __name__ == '__main__':
   main()





