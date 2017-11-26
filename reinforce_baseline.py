from utilities import *

class Model:
    def __init__(self, observation_space, action_space, learning_rate, discount_rate):
        self._action = None
        self._reward = None
        self._input = None
        self._policy = None
        self._value = None
        self._train_policy = None
        self._train_value = None
        self._t = None
        self.session = tf.Session()
        self.build_graph(observation_space, action_space, learning_rate, discount_rate)

    def build_graph(self, observation_space, action_space, learning_rate, discount_rate):
        self._action = tf.placeholder(tf.uint8)
        self._reward = tf.placeholder(tf.float32)
        self._t = tf.placeholder(tf.float32)

        width = observation_space[0]
        height = observation_space[1]

        self._input = tf.placeholder(tf.float32, [width, height, 1])
        inputs_reshaped = tf.reshape(self._input, [1, width * height])
        self._value = fully_connected(inputs_reshaped, "value", 1)

        logits = fully_connected(inputs_reshaped, "policy", action_space)
        self._policy = tf.nn.softmax(logits)
        action_mask = tf.one_hot(self._action, action_space)
        eligibility_vector = action_mask * tf.log(self._policy + 1e-13)
        parameters_policy = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='policy')
        gradients_policy = tf.gradients(eligibility_vector, parameters_policy)
        delta_policy = parameters_policy[0] + learning_rate * tf.pow(discount_rate, self._t) * (self._reward - self._value) * gradients_policy[0]
        self._train_policy = parameters_policy[0].assign(delta_policy)

        # value
        parameters_value = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='value')
        gradients_value = tf.gradients(self._value, parameters_value)
        delta_value = parameters_value[0] + learning_rate * (self._reward - self._value) * gradients_value[0]
        self._train_value = parameters_value[0].assign(delta_value)
        self.session.run(tf.global_variables_initializer())

    def train(self, input, reward, action, t):
        self.session.run([self._train_policy, self._train_value], feed_dict={
                                                        self._input: input,
                                                        self._reward: reward,
                                                        self._action: action,
                                                        self._t: t
        })

    def predict_action(self, input):
        policy = self.session.run(self._policy, feed_dict={self._input: input})
        action = action_with_policy(policy)
        return action