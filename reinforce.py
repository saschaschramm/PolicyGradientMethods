from utilities import *

class Model:
    def __init__(self, observation_space, action_space, learning_rate):
        self.action = None
        self.reward = None
        self.input = None
        self.policy = None
        self.train_policy = None
        self.session = tf.Session()
        self.build_graph(observation_space, action_space, learning_rate)

    def build_graph(self, observation_space, action_space, learning_rate):
        self.action = tf.placeholder(tf.uint8)
        self.reward = tf.placeholder(tf.float32)

        width = observation_space[0]
        height = observation_space[1]

        self.input = tf.placeholder(tf.float32, [width, height])
        inputs_reshaped = tf.reshape(self.input, [1, width * height])
        logits = fully_connected(inputs_reshaped, "policy", action_space)
        self.policy = tf.nn.softmax(logits)

        action_mask = tf.one_hot(self.action, action_space)
        eligibility_vector = action_mask * tf.log(self.policy + 1e-13)
        parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')

        gradients = tf.gradients(eligibility_vector, parameters)
        delta = (parameters[0] + learning_rate * self.reward * gradients[0])
        self.train_policy = parameters[0].assign(delta)
        self.session.run(tf.global_variables_initializer())

    def train(self, input, reward, action):
        self.session.run(self.train_policy, feed_dict={self.input: input,
                                                        self.reward: reward,
                                                        self.action: action})

    def predict_action(self, input):
        policy = self.session.run(self.policy, feed_dict={self.input: input})
        return action_with_policy(policy)

