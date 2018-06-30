from common.utilities import *

class Model:
    def __init__(self, observation_space, action_space, learning_rate):
        self.session = tf.Session()
        self.build_graph(observation_space, action_space, learning_rate)

    def build_graph(self, observation_space, action_space, learning_rate):
        self.action = tf.placeholder(tf.uint8, name="action")
        self.reward = tf.placeholder(tf.float32, name="reward")
        self.observation = tf.placeholder(tf.uint8, name="observation")

        logits = fully_connected(input=[tf.one_hot(self.observation, observation_space)],
                                 scope="policy",
                                 in_size=observation_space,
                                 out_size=action_space)

        self.policy = tf.nn.softmax(logits)

        action_mask = tf.one_hot(self.action, action_space)
        eligibility_vector = action_mask * tf.log(self.policy + 1e-13)
        parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')

        gradients = tf.gradients(eligibility_vector, parameters)
        delta = (parameters[0] + learning_rate * self.reward * gradients[0])
        self.train_policy = parameters[0].assign(delta)
        self.session.run(tf.global_variables_initializer())

    def train(self, observation, reward, action):
        self.session.run(self.train_policy, feed_dict={self.observation: observation,
                                                        self.reward: reward,
                                                        self.action: action})

    def predict_action(self, observation):
        policy = self.session.run(self.policy, feed_dict={self.observation: observation})
        return action_with_policy(policy)