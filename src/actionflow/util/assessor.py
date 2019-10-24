import numpy as np

from ..util import DLogger


class Assessor:
    def __init__(self):
        pass

    @staticmethod
    def evaluate_fit(policies, choices, random_breaktie=False, normalize=True):

        if random_breaktie and policies.shape[1] != 2:
            DLogger.logger().warning("Cannot perform random tie break for no action != 2")

        if random_breaktie and policies.shape[1] == 2:
            p = Assessor.break_ties(policies, 0.00000001)
        else:
            p = np.argmax(policies, axis=1)
        if normalize:
            return float((p == choices).sum()) / choices.shape[0]
        else:
            return float((p == choices).sum())


    @staticmethod
    def evaluate_fit_multi(policies, test, pol_in_log=False, random_tie=True):
        ids = policies['id'].unique()
        n_actions = policies.shape[1]-3
        total_corr = 0.0
        total_sum = 0.0
        total_nlp = 0
        for id in ids:
            sub_c = np.array(test.loc[test.id == id]['action']).astype(np.int)
            sub_p = np.array((policies.loc[policies.id == id]).ix[:, 1:(n_actions + 1)])
            sub_p = sub_p[sub_c != -1]
            sub_c = sub_c[sub_c != -1]

            if not pol_in_log:
                sub_p = np.log(sub_p)

            total_corr += Assessor.evaluate_fit(sub_p, sub_c, random_breaktie=random_tie, normalize=False)
            total_sum += sub_c.shape[0]
            total_nlp += -sub_p[np.arange(sub_c.shape[0]), sub_c].sum()

        return total_corr / total_sum, total_nlp / total_sum, total_nlp

    @staticmethod
    def break_ties(policies, tol):
        a1 = np.isclose(policies[:, 0], np.max(policies, axis=1), tol)
        a2 = np.isclose(policies[:, 1], np.max(policies, axis=1), tol)

        unnorm_p = np.array([a1, a2])

        p = unnorm_p * 1.0 / unnorm_p.sum(axis=0)

        return Assessor.random_choices(p.T)


    @staticmethod
    def random_choices(p):
        c = p.cumsum(axis=1)
        u = np.random.rand(len(c), 1)
        return (u < c).argmax(axis=1)