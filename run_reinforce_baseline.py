from models.reinforce_baseline import *
from common.utilities import discount
from common.stats_recorder import StatsRecorder

def main():
    global_seed(0)
    discount_rate = 0.99
    env = init_env()
    model = Model(observation_space=16,
                  action_space=4,
                  learning_rate=0.1,
                  discount_rate=discount_rate)

    stats_recorder = StatsRecorder(summary_frequency=10000, performance_num_episodes=100)

    observations = []
    rewards = []
    actions = []

    timesteps = 100000
    observation = env.reset()

    for t in range(timesteps):

        action = model.predict_action(observation)
        observations.append(observation)
        observation, reward, done, _ = env.step(action)
        rewards.append(reward)
        actions.append(action)
        stats_recorder.after_step(reward=reward, done=done, t=t)

        if done:
            i = 0
            for observation, reward, action in zip(observations, rewards, actions):
                discounted_reward = discount(rewards, discount_rate, i)
                model.train(observation, discounted_reward, action, i)
                i += 1

            observations = []
            rewards = []
            actions = []
            observation = env.reset()

if __name__ == '__main__':
   main()