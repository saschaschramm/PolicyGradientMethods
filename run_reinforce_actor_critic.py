from models.reinforce_actor_critic import *
from common.stats_recorder import StatsRecorder

def main():
    raise NotImplementedError
    # Actor Critic algorithm not working yet

    global_seed(0)
    discount_rate = 0.99
    env = init_env()
    model = Model(observation_space=16,
                  action_space=4,
                  learning_rate=0.01)

    stats_recorder = StatsRecorder(summary_frequency=10000, performance_num_episodes=1000)

    I = 1
    observation = env.reset()
    last_value = model.predict_value(observation)
    timesteps = 100000

    for t in range(timesteps):
        action = model.predict_action(observation)  # S_t
        next_observation, reward, done, _ = env.step(action)  # S_t+1, R_t+1

        stats_recorder.after_step(reward=reward, done=done, t=t)
        value = model.predict_value(next_observation)  # v(S_t+1)

        td_target = reward + discount_rate * value
        td_error = td_target - last_value
        model.train(observation, td_error, action, I)

        observation = next_observation
        last_value = value
        I = discount_rate * I

        if done:
            I = 1
            observation = env.reset()
            last_value = model.predict_value(observation)

if __name__ == '__main__':
   main()