from numpy.random import choice
import numpy as np
from scipy.misc import logsumexp


class RL:

    def beh_feed(self, actions, rewards, init_values=None):
        raise Exception('method not implemented')

    def simulate(self, sess, reward, choices, init_values=None):
        raise Exception('method not implemented')

    def get_params(self):
        raise Exception('method not implemented')

    def set_params(self, params):
        raise Exception('method not implemented')

    def _simulate_(self):
        raise Exception('method not implemented')

    def ell(self):
        return self._simulate_()[2]

    def get_trainables(self):
        raise Exception('method not implemented')

    def get_obj(self):
        raise Exception('method not implemented')

    def get_policy(self):
        raise Exception('method not implemented')

    def sim_output(self, sess, rewards, actions, pre_step_vars):
        raise Exception('method not implemented')

    def init_sim_vars(self):
        raise Exception('method not implemented')

    def get_transformed_params(self):
        raise Exception('method not implemented')

    def get_trans_func(self):
        raise Exception('method not implemented')

    def simulate_env(self, sess, max_time, env_model, greedy=False):
        policy_lists = []
        r_lists = []
        state_lists = []
        actions_list = []

        s, r, prev_action = env_model(None, None, -1)
        pre_step_vars = self.init_sim_vars()
        for t in range(max_time + 1):
            s, r, next_a = env_model(s, prev_action, t)

            rewards = np.array([[r]])
            actions = np.array([[prev_action]])

            a_dist, pre_step_vars = self.sim_output(sess, rewards, actions, pre_step_vars)

            policy_lists.append(a_dist[0])
            r_lists.append(r)
            state_lists.append(s)
            actions_list.append(prev_action)

            if next_a is not None:
                a = next_a
            elif greedy:
                a = np.argmax(a_dist[0,0])
            else:
                a = choice(np.arange(a_dist[0,0].shape[0]), p=np.exp(a_dist[0,0]))

            prev_action = a

        # note that policy is for the next trial
        return np.hstack(state_lists)[:-1], \
               np.vstack(policy_lists)[:-1, ], \
               np.hstack(r_lists)[1:], \
               np.hstack(actions_list)[1:]

    def get_param_names(self):
        raise Exception('method not implemented')

    @staticmethod
    def generate_actions(ps, n_choices, n_actions, alpha, gamma, init_v):
        v = init_v
        rewards = np.zeros((n_choices, 1))
        actions_onehot = np.zeros((n_choices, n_actions))
        actions = []
        for i in range(n_choices):
            policy = np.exp(v * gamma) / np.exp(logsumexp(gamma * v))
            action = np.argmax(np.random.multinomial(1, policy, size=1))
            actions.append(action)
            actions_onehot[i, action] = 1
            reward = np.random.binomial(1, ps[action])
            v[action] += alpha * (reward - v[action])
            rewards[i, :] = reward
        return actions_onehot, np.array(actions).astype(int), rewards
