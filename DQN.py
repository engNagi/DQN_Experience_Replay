import os
import tensorflow as tf
import numpy as np
import random


class DQN(object):
    def __init__(self, learning_rate, network_name, actions_num, state_size=(84, 84, 4), Dim_fuly_conected_lyr=512, save_dir="models/DQN/"):
        self.network_name = network_name
        self.save_dir = save_dir
        self.learning_rate = learning_rate
        self.actions_num = actions_num
        self.state_size = state_size
        # Size of number of unit in the fully connected layers
        self.Dim_fuly_conected_lyr = Dim_fuly_conected_lyr
        self.checkpoint_file = os.path.join(save_dir, 'Deep_QNetwork.ckpt')
        # Keep track of Trainable variable in the network with corresponding name
        # to copy one network to another

        with tf.variable_scope(self.network_name):
            # variable for the input to the both networks which is the states
            self._input_states = tf.placeholder(tf.float32, [None, *self.state_size], name="input_states")

            # predicted action by the Agent
            self.actions_ = tf.placeholder(tf.int32, [None,], name="actions_")

            # Target-Q is considered to approximated to "R(s,a) + Gamma * Max{Q(s´, a´ , w)}"
            # TODO Reward normalizing, Statics Simulation
            # TODO polyak Avergaing 2000 1/20000-> Done,
            # TODO Cut the image, Image skipping  -> Later
            self._Q_target = tf.placeholder(tf.float32, [None, ], name="Target_Q_Values")

            #####IMPORTANT##############
            """When initializing a deep network, it is in principle advantageous to keep the scale of the input variance constant, 
            so it does not explode or diminish by reaching the final layer.
             To be Tried
             Xavier initialization -> sets a layer’s weights to values chosen from a random uniform distribution that’s bounded between
             +-(sqrt(6)/sqrt(ni + ni+1)) where nᵢ is the number of incoming network connections, or “fan-in,” to 
             the layer, and nᵢ₊₁ is the number of outgoing network connections from that layer, also known as the “fan-out.”
            """
            ####Convolution network#####
            #1st CONV_LAYER
            first_conv_layer = tf.layers.conv2d(self._input_states, filters=32,
                                                kernel_size=(8, 8), strides=4, name="1st_conv_layer",
                                                kernel_initializer=tf.variance_scaling_initializer(scale=2))
            activated_first_conv_layer = tf.nn.relu(first_conv_layer)

            # 2nd CONV_LAYER
            second_conv_layer = tf.layers.conv2d(activated_first_conv_layer, filters=64,
                                                 kernel_size=(4, 4), strides=2, name="2nd_conv_layer",
                                                 kernel_initializer=tf.variance_scaling_initializer(scale=2))
            activated_second_conv_layer = tf.nn.relu(second_conv_layer)

            #3rd CONV_LAYER
            third_conv_layer = tf.layers.conv2d(activated_second_conv_layer, filters=64,
                                                kernel_size=(3, 3), strides=1, name="3rd_conv_layer",
                                                kernel_initializer=tf.variance_scaling_initializer(scale=2))
            activated_third_conv_layer = tf.nn.relu(third_conv_layer)

            ####Flatten_CONV_LAYERS_OUTPUT####
            flatten_cov = tf.layers.flatten(activated_third_conv_layer, name="flattened_output")

            ####DENSE_HIDDEN_LAYER######
            first_dense_layer = tf.layers.dense(flatten_cov, units=self.Dim_fuly_conected_lyr,
                                                activation=tf.nn.relu,
                                                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                                name="1st_Dense_layer")

            #####OUTPUT LAYER_is the predicted action of the Network#########
            #number of units equal to the number of action_space
            self._predicted_action = tf.layers.dense(first_dense_layer, units=self.actions_num,
                                                        kernel_initializer=tf.variance_scaling_initializer(scale=2),
                                                        name="Actions")

            ##### Action Q-values_one hot encoding#####
            self._actions_q_value = tf.reduce_sum(tf.multiply(self._predicted_action,
                                                  tf.one_hot(self.actions_, self.actions_num)),
                                                  reduction_indices=[1])

            ######COST= Difference between calculated Q-values by the model and the Target Q-values#########
            self._cost = tf.reduce_mean(tf.square(self._Q_target - self._actions_q_value))


            #TODO Try differnet optimization techniques like RMS, Gradient_Descent
            """Adam_optimizer ->  computes individual adaptive learning rates for different parameters from estimates of
                                first and second moments of the gradients"""
            ####TRAINING####
            self.optimization = tf.train.AdamOptimizer(self.learning_rate).minimize(self._cost)

    def create_session(self, session):
        self.session = session

    # calculation Action-Q-Values for the given input state
    def predict(self, states):
        states = np.moveaxis(states, 1, -1)
        return self.session.run(self._predicted_action, feed_dict={self._input_states: states})

    #input the target, states, actions from the eviroment to update the weights(Qtheta) of Q(s,a|Qtheta)
    def training(self, targets, states, actions):
        states = np.moveaxis(states, 1, -1)
        cost, _ = self.session.run([self._cost, self.optimization],
                                feed_dict={self.actions_: actions,
                                           self._input_states: states,
                                           self._Q_target: targets})
        return cost

    def eps_explore(self, epsilon, state):
        if np.random.random() < epsilon:
            return np.random.choice(self.actions_num)
        else:
            return np.argmax(self.predict([state])[0])

    # Soft update: polyak averaging.
    def polyek_target_n_update(self, mine, other, tau=0.05):
        # Get all the variables in the Q primary network.
        q_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="other")
        # Get all the variables in the Q target network.
        q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="mine")
        assert len(q_vars) == len(q_target_vars)

        # q_target_vars = [t for t in tf.trainable_variables() if t.name.startswith(mine)]
        # q_target_vars = sorted(q_target_vars, key=lambda v: v.name)
        # q_vars = [t for t in tf.trainable_variables() if t.name.startswith(other)]
        # q_vars = sorted(q_vars, key=lambda v: v.name)

        # ops = []
        # for q_target_vars, q_vars in zip(q_target_vars, q_vars):
        #     actual = self.session.run(q_vars)
        #     op = q_target_vars.assign(actual)
        #     ops.append(op)

        self.session.run([v_t.assign(v_t * (1. - tau) + v * tau) for v_t, v in zip(q_target_vars, q_vars)])

    def learn(self, model, target_model, experience_replay_buffer, gamma, batch_size):
        # Sample experiences
        samples = random.sample(experience_replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*samples))
        # Calculate targets
        next_Qs = target_model.predict(next_states)
        next_Q = np.amax(next_Qs, axis=1)
        targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q
        # Update model
        loss = model.training(targets, states, actions)
        return loss

    def chk_pnt_load(self):
        self.saver = tf.train.Saver(tf.global_variables())
        load_was_successful = True

        try:
            print("...Loading check point...")
            chk_pnt = tf.train.get_checkpoint_state(self.checkpoint_file)
            load_dir = chk_pnt.model_checkpoint_path
            self.saver.restore(self.session, load_dir)
        except:
            print("...No saved model to Load...")
            load_was_successful = False
        else:
            print("loaded model :{}".format(load_dir))
            saver = tf.train.Saver(tf.global_variables())

    def chk_pnt_save(self, n):
        print("...Saving check point...")
        self.saver.save(self.session, self.checkpoint_file, global_step=n)
        print("Model Saved # {}".format(n))

