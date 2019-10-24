import numpy as np

def bandit_evn(p1, p2, init_state=0, init_action=0, init_reward=0):
    def env(s, a, trial):

        # initial state action
        if trial == -1:
            return init_state, init_reward, init_action

        rnd = np.random.uniform(0, 1, 1)
        if a == 0:
            if rnd < p1:
                return s, 1, None
            return s, 0, None

        if a == 1:
            if rnd < p2:
                return s, 1, None
            return s, 0, None

        if a == -1:
            return s, 0, None

    return env
