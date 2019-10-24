# Different kinds of environments for the simulation of the model

def fixed_env(a_cut, reward_trial, init_state=0, init_action=0, init_reward=0):
    def env(s, a, trial):

        # initial state action
        if trial == -1:
            return init_state, init_reward, init_action

        next_a = 0
        r = 0

        if trial < a_cut:
            next_a = 0

        if trial >= a_cut:
            next_a = 1

        if trial in reward_trial:
            r = 1

        return s, r, next_a

    return env


def a2_generic(a1_period, off_pol_trials, reward_trials, init_state=0, init_action=0, init_reward=0):
    def env(s, a, trial):

        # initial state action
        if trial == -1:
            return init_state, init_reward, init_action

        if off_pol_trials(trial):

            if a1_period(trial):
                next_a = 0
            else:
                next_a = 1

        else:
            next_a = None

        if reward_trials(trial):
            r = 1
        else:
            r = 0

        return s, r, next_a

    return env


def fix_env_oss(init_state=0, init_action=0, init_reward=0):
    def env(s, a, trial):

        #initial state action
        if trial == -1:
            return init_state, init_reward, init_action

        if trial < 3:
            return s, 0, 0

        if trial == 3:
            return s, 0, 1

        if trial == 4:
            return s, 0, 0

        if trial == 5:
            return s, 0, 1

        return s, 0, None
    return env


def fix_env2(a_period, r_period, init_state=0, init_action=0, init_reward=0):
    def env(s, a, trial):

        #initial state action
        if trial == -1:
            return init_state, init_reward, init_action

        if int(trial / a_period) % 2 == 0:
            next_a = 0
        else:
            next_a = 1

        if trial % r_period == 0:
            r = 0
        else:
            r = 1

        return s, r, next_a

    return env