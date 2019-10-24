# Simulates QL, QLP and GQL on policy.

import tensorflow as tf
from actionflow.qrl.gql import GQL
from actionflow.qrl.opt_ml import OptML
from actionflow.qrl.ql import QL
from actionflow.qrl.simulate import Simulator
from actionflow.sim.envs import bandit_evn
from actionflow.util import DLogger
from actionflow.util.helper import id_generator, load_model
from BD.sim.sims import get_BD_confs
from BD.util.paths import Paths
import os
import pandas as pd


def simulate_ml():

    _, _, by_group = get_BD_confs()

    for g in by_group.keys():
        subj_list = by_group[g]

        tf.reset_default_graph()
        cur_group = 'gql10d-ml-opt/' + g
        model_path = Paths.rest_path + 'archive/beh/' + cur_group + '/model-final/'
        output_prefix = Paths.local_path + 'BD/sims/gql10d-ml/'
        worker = GQL.get_instance(2, 10, {})

        DLogger.logger().debug("model path: " + model_path)
        DLogger.logger().debug("output prefix" + output_prefix)

        onpol_sim(worker, subj_list, output_prefix, model_path)

    for g in by_group.keys():
        subj_list = by_group[g]

        tf.reset_default_graph()
        cur_group = 'qlp-ml-opt/' + g
        model_path = Paths.rest_path + 'archive/beh/' + cur_group + '/model-final/'
        output_prefix = Paths.local_path + 'BD/sims/qlp-ml/'
        worker = QL.get_instance_with_pser(2, 0.1, 0.1, 0.1)

        DLogger.logger().debug("model path: " + model_path)
        DLogger.logger().debug("output prefix" + output_prefix)

        onpol_sim(worker, subj_list, output_prefix, model_path)

    for g in by_group.keys():
        subj_list = by_group[g]

        tf.reset_default_graph()
        cur_group = 'ql-ml-opt/' + g
        model_path = Paths.rest_path + 'archive/beh/' + cur_group + '/model-final/'
        output_prefix = Paths.local_path + 'BD/sims/ql-ml/'
        worker = QL.get_instance_without_pser(2, 0.1, 0.1)

        DLogger.logger().debug("model path: " + model_path)
        DLogger.logger().debug("output prefix" + output_prefix)

        onpol_sim(worker, subj_list, output_prefix, model_path)

    for g in by_group.keys():
        subj_list = by_group[g]

        tf.reset_default_graph()
        cur_group = 'gql-ml-opt/' + g
        model_path = Paths.rest_path + 'archive/beh/' + cur_group + '/model-final/'
        output_prefix = Paths.local_path + 'BD/sims/gql-ml/'
        worker = GQL.get_instance(2, 2, {})

        DLogger.logger().debug("model path: " + model_path)
        DLogger.logger().debug("output prefix" + output_prefix)

        onpol_sim(worker, subj_list, output_prefix, model_path)


def onpol_sim(worker, subj_list, output_prefix, model_path):
    params = OptML.get_variables(worker.get_params())
    worker.set_params(params)

    config = tf.ConfigProto(device_count={'GPU': 0})
    with tf.Session(config=config) as sess:

        DLogger.logger().debug("loading mode....")
        load_model(sess, model_path)
        DLogger.logger().debug("finished loading mode.")
        for s in range(len(subj_list)):
            DLogger.logger().debug("parameters {}".format(sess.run(params)))
            for c in subj_list[s]:
                DLogger.logger().debug("subject {} trial {}".format(c['id'], c['block']))
                choices = c['choices']
                output_path = output_prefix + 'sim_' + id_generator(size=7) + '/'

                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                c2 = c.copy()
                c2['option'] = {}
                c2['N'] = c['id']
                pdc = pd.DataFrame(c2, index=[0])
                pdc.to_csv(output_path + 'config.csv', index=False)

                _, _ = Simulator.simulate_env(sess, worker,
                                              output_path,
                                              choices,
                                              bandit_evn(c['prop0'], c['prop1'], init_state=None,
                                                         init_action=-1,
                                                         init_reward=0),
                                              greedy=False
                                              )


if __name__ == '__main__':
    simulate_ml()
