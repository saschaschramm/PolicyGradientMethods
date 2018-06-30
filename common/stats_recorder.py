class StatsRecorder:

    def __init__(self, summary_frequency, performance_num_episodes):
        self.total_rewards = []
        self.summary_frequency = summary_frequency
        self.performance_num_episodes = performance_num_episodes
        self.total_reward = 0
        self.num_episodes = 0

    def print_score(self, t):
        score = sum(self.total_rewards[-self.performance_num_episodes:]) / self.performance_num_episodes
        print("{} {}".format(t, score))

    def after_step(self, reward, done, t):
        self.total_reward += reward

        if t % self.summary_frequency == 0:
            self.print_score(t)

        if done:
            self.num_episodes += 1
            self.total_rewards.append(self.total_reward)
            self.total_reward = 0