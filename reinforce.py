from utilities import *

class Model:
    def __init__(self, observation_space, action_space, learning_rate):
        self._action = None
        self._reward = None
        self._input = None
        self._policy = None
        self._value = None
        self._train_policy = None
        self.session = tf.Session()
        self.build_graph(observation_space, action_space, learning_rate)

    def build_graph(self, observation_space, action_space, learning_rate):
        self._action = tf.placeholder(tf.uint8)
        self._reward = tf.placeholder(tf.float32)

        width = observation_space[0]
        height = observation_space[1]

        self._input = tf.placeholder(tf.float32, [width, height, 1])
        inputs_reshaped = tf.reshape(self._input, [1, width * height])

        logits = fully_connected(inputs_reshaped, "policy", action_space)
        self._policy = tf.nn.softmax(logits)

        action_mask = tf.one_hot(self._action, action_space)
        eligibility_vector = action_mask * tf.log(self._policy + 1e-13)
        parameters = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')

        gradients = tf.gradients(eligibility_vector, parameters)
        delta = (parameters[0] + learning_rate * self._reward * gradients[0])
        self._train_policy = parameters[0].assign(delta)
        self.session.run(tf.global_variables_initializer())

    def train(self, input, reward, action):
        self.session.run(self._train_policy, feed_dict={self._input: input,
                                                        self._reward: reward,
                                                        self._action: action})

    def predict_action(self, input):
        policy = self.session.run(self._policy, feed_dict={self._input: input})
        return action_with_policy(policy)

