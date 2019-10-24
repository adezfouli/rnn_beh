import tensorflow as tf
import numpy as np
from actionflow.rnn.rnn_base import RNNBase
from actionflow.util import DLogger
from ..rnn.consts import Const
from numpy.random import choice


class GRUBase(RNNBase):
    def __init__(self, a_size, s_size, n_cells):

        # a dummy variable for calculating gradients wrt to cell state in - it is set in run_rnn method
        self.state_in_pret = None

        RNNBase.__init__(self, a_size, s_size, n_cells)

    def rnn_cells(self, n_cells, rnn_in):
        # RNN component
        gru = tf.contrib.rnn.GRUCell(n_cells)

        # init input to the model
        h_init = np.zeros((1, gru.state_size), np.float32)
        state_init = h_init

        # init state of the RNN
        h_in = tf.placeholder(Const.FLOAT, [None, gru.state_size])
        state_in = h_in

        step_size = tf.ones([tf.shape(self.prev_actions)[0]], dtype=tf.int32) * tf.shape(self.prev_actions)[1]

        rnn_out, state_out, gru_state_track = self.run_rnn(None, h_in, gru, n_cells, rnn_in, step_size)
        # DIM rnn_out: nBatches x (nChoices + 1) x nCells
        # DIM state_out: 1 * nBatches x nCells
        # DIM gru_state_track: nBatches x 1 x (nChoices + 1) x nCells

        return state_init, state_in, rnn_out, state_out, gru_state_track

    def run_rnn(self, c_in, h_in, gru_cell, n_cells, rnn_in, step_size):

        self.state_in_pret = tf.zeros([self.n_batches, step_size[0], self.n_cells], dtype=Const.FLOAT)

        def learn(curr_trial, state_track, output_track):
            state_in = state_track[:, -1, :]
            input_in = rnn_in[:, curr_trial, :]
            gru_outputs, gru_state = tf.nn.dynamic_rnn(
                gru_cell, input_in[:, np.newaxis, :], initial_state=state_in, time_major=False)

            return tf.add(curr_trial, 1), \
                   tf.concat((state_track, gru_state[:, np.newaxis, :] + self.state_in_pret[:, curr_trial, np.newaxis, :]), axis=1),\
                   tf.concat((output_track, gru_outputs + self.state_in_pret[:, np.newaxis, curr_trial, :]), axis=1)

        def cond(curr_trial, _1, _2):
            return tf.less(curr_trial, step_size[0])

        while_output = tf.while_loop(cond, learn,
                                     loop_vars=[tf.constant(0),
                                                h_in[:, np.newaxis],
                                                tf.zeros((self.n_batches, 1, n_cells), dtype=Const.FLOAT)
                                                ],
                                     shape_invariants=[tf.constant(0).get_shape(),
                                                       tf.TensorShape([None, None, n_cells]),
                                                       tf.TensorShape([None, None, n_cells])
                                                       ]
                                     )
        _, state_track, output_track = while_output

        # remove the first dummy variables
        rnn_out = output_track[:, 1:]
        gru_h = state_track[:, -1, :]
        state_out = (gru_h)

        # remove the first variable
        gru_state_track = state_track[:, np.newaxis, 1:]

        return rnn_out, state_out, gru_state_track

    # this one runs using TF builtin RNN runner, however, it does not
    # exposre track of c and h - so it is reimplemented
    # def run_rnn(self, c_in, h_in, lstm_cell, n_cells, rnn_in, step_size):
    #     state_in = h_in
    #     gru_out, gru_state = tf.nn.dynamic_rnn(
    #         lstm_cell, rnn_in, initial_state=state_in,
    #         time_major=False, sequence_length=step_size)
    #     state_out = gru_state
    #     rnn_out = gru_out
    #     return rnn_out, state_out, rnn_out[:, np.newaxis]

    def dPolicy_dH(self, policy_trials):
        """
        Calculates the gradient of policies at policy_trials wrt to rnn_outs (for all trials)
        :param policy_trials: the array/list containing index of the action for which the gradinet of the cell
                output is calculated.
        :return: an array of size nCells x len(policy_trials) x nTrials
        """

        DLogger.logger().debug("adding gradients for each cell-action pair...")
        grads = []
        for a in policy_trials:
            grads.append(tf.gradients(self.policy[0, a, 0], self.state_in_pret)[0][0])
        grads = tf.convert_to_tensor(grads)
        grads = tf.transpose(grads, [2, 0, 1])
        DLogger.logger().debug("finished adding gradients.")
        return grads

    def dPolicy_dH_dR(self, policy_trials, H_trials):
        """
        :return: an array of size nCells x len(H_trials) x nTrials (corresponding to rewards) x len(policy_trials)
        """

        dPdH = self.dPolicy_dH(policy_trials)
        dHdR = self.dH_dReward(H_trials)

        dPdH = tf.gather(dPdH, H_trials, axis=2)
        # equivalent to this numpy dPdH = dPdH[:, :, rnn_out_trials]

        dPold_dH_dR = (dHdR[:, :, :, np.newaxis] * tf.transpose(dPdH, [0, 2, 1])[:, :, np.newaxis, :])
        return dPold_dH_dR

    def beh_feed(self, actions, rewards, states):
        """
        Created a dict for TensorFlow by adding a dummy action and reward to the beginning of
        actions and rewards and a dummy state to the end of states
        """

        prev_rewards = np.hstack((np.zeros((rewards.shape[0], 1)), rewards))
        prev_actions = np.hstack((-1 * np.ones((actions.shape[0], 1)), actions))
        rnn_state = self.state_init
        feed_dict = {self.prev_rewards: prev_rewards[:, :, np.newaxis],
                     self.prev_actions: prev_actions,
                     self.actions: actions,
                     self.state_in: np.repeat(rnn_state, rewards.shape[0], axis=0)}
        if states is not None:
            prev_states = np.hstack((states, np.zeros(states[:, 0:1].shape)))
            feed_dict[self.prev_states] = prev_states
        return feed_dict

    def get_h_track(self):
        # note we ignore the first element since there is one extra action added at the beginning
        return tf.transpose(self.state_track[:, 0, 1:, :], [0, 2, 1])

    def get_c_track(self):
        return tf.transpose(self.state_track[:, 0, 1:, :], [0, 2, 1])

    def simulate_env(self, sess, max_time, env_model, greedy=False, cell_mask=None):
        rnn_state_lists = []
        policy_lists = []
        r_lists = []
        state_lists = []
        actions_list = []

        if cell_mask is None:
            cell_mask = np.ones((1, self.state_in.shape[1]))

        rnn_state = self.state_init
        s, r, a = env_model(None, None, -1)
        for t in range(max_time + 1):
            s, r, next_a = env_model(s, a, t)
            feed_dict = {
                self.prev_rewards: [[[r]]],
                self.timestep: [[t]],
                self.prev_actions: [[a]],
                self.state_in: rnn_state,
                }

            if s is not None:
                feed_dict[self.prev_states] = [[s]]

            a_dist, rnn_state_new = sess.run([self.policy, self.state_out], feed_dict=feed_dict)
            rnn_state_new = rnn_state_new * cell_mask
            rnn_state = rnn_state_new
            rnn_state_lists.append(rnn_state_new)
            policy_lists.append(a_dist)
            r_lists.append(r)
            state_lists.append(s)
            actions_list.append(a)

            if next_a is not None:
                a = next_a
            elif greedy:
                a = np.argmax(a_dist)
            else:
                a = choice(np.arange(a_dist.shape[2]), p=a_dist[0, 0])

        # note that policy is for the next trial
        return np.hstack(state_lists)[:-1], \
               np.hstack(policy_lists)[0, :-1], \
               np.hstack(r_lists)[1:], \
               np.hstack(actions_list)[1:], \
               np.vstack(rnn_state_lists)[np.newaxis, 1:, ]



