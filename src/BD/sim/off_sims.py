# Performs off-policy simulations

import tensorflow as tf
from actionflow.qrl.gql import GQL
from actionflow.qrl.opt_ml import OptML
from actionflow.qrl.ql import QL
from actionflow.rnn.lstm_beh import LSTMBeh
from actionflow.rnn.simulate import Simulator
from actionflow.util.helper import load_model
from BD.sim.envs import fixed_env,  a2_generic
from BD.util.paths import Paths
import numpy as np


def simulate_BD_RNN(input_path, output_path):
    model_iter = 'model-final'
    conds = [{'name': '1R1L', 'rewards': [4, 27]},
             {'name': '1R2L', 'rewards': [4, 22, 27]},
             {'name': '1R3L', 'rewards': [4, 14, 22, 27]},
             ]

    groups = [
        {'group': 'Healthy', 'cells': 10},
        {'group': 'Bipolar', 'cells': 20},
        {'group': 'Depression', 'cells': 10}
    ]

    for g in groups:
        for cond in conds:
            tf.reset_default_graph()
            worker = LSTMBeh(2, 0, g['cells'])
            Simulator.simulate_env(worker,
                                   input_path + str(g['cells']) +
                                   'cells/' + str(g['group']) + '/' + model_iter + '/',
                                   output_path + cond['name'] + '/' + str(g['group']) + '/',
                                   30,
                                   fixed_env(10, cond['rewards'], init_state=None, init_action=-1, init_reward=0),
                                   greedy=True
                                   )


def simulate_BD_osci_RNN(input_path, output_path):

    model_iter = 'model-final'

    groups = [
        {'group': 'Healthy', 'cells': 10},
        {'group': 'Bipolar', 'cells': 20},
        {'group': 'Depression', 'cells': 10}
    ]

    for g in groups:
            tf.reset_default_graph()
            worker = LSTMBeh(2, 0, g['cells'])
            freq = 1
            baseline = 5
            Simulator.simulate_env(worker,
                                   input_path + str(g['cells']) +
                                   'cells/' + str(g['group']) + '/' + model_iter + '/',
                                   output_path + str(g['group']) + '/',
                                   20,
                                   a2_generic(lambda trial: trial < baseline or np.floor((trial - baseline) / freq) % 2 == 0,
                                              lambda trial: trial < baseline or (trial - baseline) < (freq * 2) * 2,
                                              lambda trial: False,
                                              init_state=None,
                                              init_action=-1,
                                              init_reward=0
                                              ),
                                   greedy=True
                                   )


def simulate_mix_QL(model_name, output_folder, worker):


    conds = [{'name': '1R1L', 'rewards': [4, 27]},
             {'name': '1R2L', 'rewards': [4, 22, 27]},
             {'name': '1R3L', 'rewards': [4, 14, 22, 27]},
             ]

    for group in ['Healthy', 'Bipolar', 'Depression']:
        for cond in conds:
            config = tf.ConfigProto(device_count={'GPU': 0})
            with tf.Session(config=config) as sess:
                load_model(sess, Paths.rest_path + 'archive/beh/' + model_name + '/' + group + '/model-final/')
                import actionflow.qrl.simulate as qsim
                qsim.Simulator.simulate_env(sess, worker,
                                       Paths.local_path + 'BD/sims/reward/' + output_folder + '/' + cond['name'] + '/' + group + '/',
                                       30,
                                       fixed_env(10, cond['rewards'], init_state=None, init_action=-1, init_reward=0),
                                       greedy=True
                                       )


def simulate_oscci_mix_QL(model_name, output_folder, worker):

    for group in ['Healthy', 'Bipolar', 'Depression']:
        freq = 1
        baseline = 5
        config = tf.ConfigProto(device_count={'GPU': 0})
        with tf.Session(config=config) as sess:
            load_model(sess, Paths.rest_path + 'archive/beh/' + model_name + '/' + group + '/model-final/')
            import actionflow.qrl.simulate as qsim
            qsim.Simulator.simulate_env(sess, worker,
                                        Paths.local_path + 'BD/sims/osci/' + output_folder + '/' + group + '/',
                                   20,
                                        a2_generic(lambda trial: trial < baseline or np.floor(
                                            (trial - baseline) / freq) % 2 == 0,
                                                   lambda trial: trial < baseline or (trial - baseline) < (
                                                                                                              freq * 2) * 2,
                                                   lambda trial: False,
                                                   init_state=None,
                                                   init_action=-1,
                                                   init_reward=0
                                                   ),
                                        greedy=True
                                        )


if __name__ == '__main__':
    # tf.reset_default_graph()
    # worker = QL.get_instance_without_pser(2, 0.1, 0.2)
    # group_log_sigma2, group_mu, ind_log_sigma2, ind_mu = OptMIX.get_variables(worker.get_params())
    # worker.set_params(group_mu)
    # simulate_mix_QL('ql-mix-opt', 'QL', worker)
    #
    # tf.reset_default_graph()
    # worker = QL.get_instance_with_pser(2, 0.1, 0.2, 0.2)
    # group_log_sigma2, group_mu, ind_log_sigma2, ind_mu = OptMIX.get_variables(worker.get_params())
    # worker.set_params(group_mu)
    # simulate_mix_QL('qlp-mix-opt', 'QLP', worker)
    #
    # tf.reset_default_graph()
    # worker = GQL.get_instance(2, 2, {})
    # group_log_sigma2, group_mu, ind_log_sigma2, ind_mu = OptMIX.get_variables(worker.get_params())
    # worker.set_params(group_mu)
    # simulate_mix_QL('gql-mix-opt', 'GQL', worker)
    #
    # tf.reset_default_graph()
    # worker = GQL.get_instance(2, 1, {})
    # group_log_sigma2, group_mu, ind_log_sigma2, ind_mu = OptMIX.get_variables(worker.get_params())
    # worker.set_params(group_mu)
    # simulate_mix_QL('gql1d-mix-opt', 'GQL1D', worker)
    #
    # tf.reset_default_graph()
    # worker = GQL.get_instance(2, 10, {})
    # group_log_sigma2, group_mu, ind_log_sigma2, ind_mu = OptMIX.get_variables(worker.get_params())
    # worker.set_params(group_mu)
    # simulate_mix_QL('gql10d-mix-opt', 'GQL10D', worker)
    #
    # tf.reset_default_graph()
    # worker = QL.get_instance_without_pser(2, 0.1, 0.2)
    # group_log_sigma2, group_mu, ind_log_sigma2, ind_mu = OptMIX.get_variables(worker.get_params())
    # worker.set_params(group_mu)
    # simulate_oscci_mix_QL('ql-mix-opt', 'QL', worker)
    #
    # tf.reset_default_graph()
    # worker = QL.get_instance_with_pser(2, 0.1, 0.2, 0.2)
    # group_log_sigma2, group_mu, ind_log_sigma2, ind_mu = OptMIX.get_variables(worker.get_params())
    # worker.set_params(group_mu)
    # simulate_oscci_mix_QL('qlp-mix-opt', 'QLP', worker)
    #
    # tf.reset_default_graph()
    # worker = GQL.get_instance(2, 2, {})
    # group_log_sigma2, group_mu, ind_log_sigma2, ind_mu = OptMIX.get_variables(worker.get_params())
    # worker.set_params(group_mu)
    # simulate_oscci_mix_QL('gql-mix-opt', 'GQL', worker)
    #
    # tf.reset_default_graph()
    # worker = GQL.get_instance(2, 10, {})
    # group_log_sigma2, group_mu, ind_log_sigma2, ind_mu = OptMIX.get_variables(worker.get_params())
    # worker.set_params(group_mu)
    # simulate_oscci_mix_QL('gql10d-mix-opt', 'GQL10D', worker)


    ############ ML graphs #############################
    tf.reset_default_graph()
    worker = QL.get_instance_without_pser(2, 0.1, 0.2)
    worker.set_params(OptML.get_variables(worker.get_params()))
    simulate_mix_QL('ql-ml-opt', 'QL', worker)

    tf.reset_default_graph()
    worker = QL.get_instance_with_pser(2, 0.1, 0.2, 0.2)
    worker.set_params(OptML.get_variables(worker.get_params()))
    simulate_mix_QL('qlp-ml-opt', 'QLP', worker)

    tf.reset_default_graph()
    worker = GQL.get_instance(2, 2, {})
    worker.set_params(OptML.get_variables(worker.get_params()))
    simulate_mix_QL('gql-ml-opt', 'GQL', worker)

    tf.reset_default_graph()
    worker = GQL.get_instance(2, 1, {})
    worker.set_params(OptML.get_variables(worker.get_params()))
    simulate_mix_QL('gql1d-ml-opt', 'GQL1D', worker)

    tf.reset_default_graph()
    worker = QL.get_instance_without_pser(2, 0.1, 0.2)
    worker.set_params(OptML.get_variables(worker.get_params()))
    simulate_oscci_mix_QL('ql-ml-opt', 'QL', worker)

    tf.reset_default_graph()
    worker = QL.get_instance_with_pser(2, 0.1, 0.2, 0.2)
    worker.set_params(OptML.get_variables(worker.get_params()))
    simulate_oscci_mix_QL('qlp-ml-opt', 'QLP', worker)

    tf.reset_default_graph()
    worker = GQL.get_instance(2, 2, {})
    worker.set_params(OptML.get_variables(worker.get_params()))
    simulate_oscci_mix_QL('gql-ml-opt', 'GQL', worker)

    tf.reset_default_graph()
    worker = GQL.get_instance(2, 10, {})
    worker.set_params(OptML.get_variables(worker.get_params()))
    simulate_oscci_mix_QL('gql10d-ml-opt', 'GQL10D', worker)

    simulate_BD_RNN(Paths.rest_path + 'archive/beh/rnn-opt-from-init/',
                    Paths.local_path + 'BD/sims/reward/RNN/')

    simulate_BD_osci_RNN(Paths.rest_path + 'archive/beh/rnn-opt-from-init/',
                         Paths.local_path + 'BD/sims/osci/RNN/')

    for i in range(15):
        simulate_BD_RNN(Paths.rest_path + 'archive/beh/rnn-opt-rand-init/run_' + str(i) + '/',
                        Paths.local_path + 'BD/sims/reward/RNN_' + str(i) + '/')

        simulate_BD_osci_RNN(Paths.rest_path + 'archive/beh/rnn-opt-rand-init/run_'+ str(i) + '/',
                             Paths.local_path + 'BD/sims/osci/RNN_' + str(i) + '/')


    # simulate_BD_GQ_1DL()