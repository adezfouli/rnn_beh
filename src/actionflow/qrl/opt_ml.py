import tensorflow as tf

from actionflow.data.data_process import DataProcess
from actionflow.qrl.gql import GQL
from actionflow.qrl.opt import Opt
from ..qrl.consts import Const
from ..qrl.ql import QL, Config
from ..util import DLogger
from ..util.assessor import Assessor
from ..util.export import Export
from ..util.helper import Helper, ensure_dir


class OptML:
    def __init__(self):
        pass

    @staticmethod
    def optimise(worker, output_path, train, test,
                 learning_rate=1e-2,
                 load_model_path=None,
                 global_iters=2000
                 ):

        DLogger.logger().debug("learning rate: " + str(learning_rate))
        DLogger.logger().debug("training data-points: " + str(len(train)))
        if test is not None:
            DLogger.logger().debug("test data-points: " + str(len(test)))
        else:
            DLogger.logger().debug("test data-points: " + str(None))

        trainables = OptML.get_variables(worker.get_params())

        DLogger.logger().debug("total params: " + str(Helper.get_num_trainables()))

        worker.set_params(trainables)
        apply_grads = Opt.get_apply_grads(worker.get_obj(), trainables, learning_rate)

        saver = tf.train.Saver()

        with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:

            if load_model_path is not None:
                DLogger.logger().debug("loading model from: " + str(load_model_path))
                ckpt = tf.train.get_checkpoint_state(load_model_path)
                tf.train.import_meta_graph(load_model_path + 'model.cptk.meta')
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            DLogger.logger().debug("opt started...")

            for iter in range(global_iters):
                local_obj = 0
                for v in train.values():
                    local_obj += OptML.train(worker, sess, v, apply_grads)
                DLogger.logger().debug("global iter = {:4d} total obj: {:7.4f}".format(iter, local_obj))

            DLogger.logger().debug("opt finished.")

            OptML.test_and_save("final", output_path, saver, sess, test, worker)

    @staticmethod
    def get_variables(init_params):
        with tf.variable_scope("opt_ml"):
            trainsables = [tf.Variable(param, dtype=Const.FLOAT, name='mu') for param in init_params]
        return trainsables

    @staticmethod
    def train(worker, sess, train_data, apply_grads):

        obj = 0
        for d in train_data:

            beh_feed = worker.beh_feed(d['action'], d['reward'])
            _, o = sess.run([apply_grads, worker.get_obj()], feed_dict=beh_feed)
            obj += o

        return obj

    @staticmethod
    def test_and_save(label, output_path, saver, sess, test, worker):
        policies_list = {}
        if test is not None and len(test) > 0:
            DLogger.logger().debug("started testing...")
            output = [['sid', 'pre accu', 'post accu']]
            for k, tr in test.iteritems():

                merged_data = DataProcess.merge_data({'tmp': tr})['merged'][0]
                ell, policies, _ = worker.simulate(sess, merged_data['reward'], merged_data['action'])

                policies_list[k] = {}
                post_accu = []
                for i in range(len(tr)):
                    v = tr[i]
                    pol = policies[i, :v['action'].shape[1]]
                    post_accu.append(Assessor.evaluate_fit(pol, v['action']))
                    policies_list[k][v['block']] = pol

                output.append([v['id'], None, post_accu])
            DLogger.logger().debug("finished testing.")

            if output_path is not None:
                Export.write_to_csv(output, output_path, 'accuracy.csv')
                Export.policies(policies_list, output_path, 'policies.csv')

        if output_path is not None:
            vparams = sess.run(worker.get_params())
            params = [worker.get_param_names(), vparams]
            Export.write_to_csv(params, output_path, 'params.csv')

            trans_params = [f(v) for f, v in zip(worker.get_trans_func(), worker.get_params())]

            output = sess.run(trans_params)
            params = [worker.get_param_names(), output]
            Export.write_to_csv(params, output_path, 'params-trans.csv')

        if output_path is not None and saver is not None:
            model_path = output_path + 'model-' + str(label) + '/'
            ensure_dir(model_path)
            saver.save(sess, model_path + 'model.cptk')
            train_writer = tf.summary.FileWriter(model_path)
            train_writer.add_graph(sess.graph)
            train_writer.close()

        return policies_list
