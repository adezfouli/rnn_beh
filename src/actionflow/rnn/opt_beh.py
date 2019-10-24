import numpy as np
import tensorflow as tf

from actionflow.rnn.opt import Opt
from ..util import DLogger
from ..util.assessor import Assessor
from ..util.export import Export
from ..util.helper import Helper, fix_init_all_vars, ensure_dir


class OptBEH(Opt):
    def __init__(self):
        pass

    @staticmethod
    def optimise(model, output_path, train, test,
                 learning_rate=1e-3,
                 global_iters=20000,
                 test_period=50,
                 load_model_path=None,
                 GPU=False
                 ):
        """
        Optimizes a RNN model using ``train`` and evaluated performance on ``test``.

        Args
        ----
            model: Model to be optimized. Should be sub-class of *RNNBase*
            output_path: The directory for saving the results in.
            train: Training data-set. It has the following structure:
                {'subject_ID':
                    [
                        {
                            'action': array of dim nBatches x nTimeSteps
                            'reward': array of dim nBatches x nTimeSteps
                            'state': array of dim nBatches x nTimeSteps x stateDim
                            'block': int
                            'id': string
                        }
                        ...
                        {
                         ...
                        }
                    ]
                }

            test: Testing data-set. It has the following structure:
                {'subject_ID':
                    [
                        {
                            'action': array of dim nBatches x nTimeSteps
                            'reward': array of dim nBatches x nTimeSteps
                            'state': array of dim nBatches x nTimeSteps x stateDim
                            'block': int
                            'id': string
                        }
                        ...
                        {
                            ....
                        }
                    ]
                }
            learning_rate: double. Learning rate used in optimizer
            global_iters: int. Number of iterations over the whole dataset.
            test_period: int. Number of optimization iterations between tests.
            load_model_path: string. Path to the directory for initializing the model from.
            GPU: boolean. Whether to use GPU.

        Returns
        -------
            Predictions made on *test* data-set.
        """
        DLogger.logger().debug("learning rate: " + str(learning_rate))
        DLogger.logger().debug("global iters: " + str(global_iters))
        DLogger.logger().debug("training data-points: " + str(len(train)))
        if test is not None:
            DLogger.logger().debug("test data-points: " + str(len(test)))
        else:
            DLogger.logger().debug("test data-points: " + str(None))

        trainables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        apply_grads = Opt.get_apply_grads(model.get_obj(), trainables, learning_rate)
        saver = tf.train.Saver(max_to_keep=None)

        if not GPU:
            config = tf.ConfigProto(device_count={'GPU': 0})
        else:
            config = tf.ConfigProto()

        test_and_save = {}

        with tf.Session(config=config) as sess:
            if load_model_path is not None:
                DLogger.logger().debug("loading model from: " + str(load_model_path))
                ckpt = tf.train.get_checkpoint_state(load_model_path)
                tf.train.import_meta_graph(load_model_path + 'model.cptk.meta')
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())


            DLogger.logger().debug("opt started...")

            for iter in range(global_iters):

                if iter % test_period == 0:
                    test_and_save[iter] = OptBEH.test_and_save(iter, output_path, saver, sess, test, model)

                ######### training #############################
                local_obj = 0
                for k in sorted(train.keys()):
                    local_obj += OptBEH.train(model, sess, train[k], apply_grads)
                #################### end of training ############

                DLogger.logger().debug("global iter = {:4d} total obj: {:7.4f}".format(iter, local_obj))

            DLogger.logger().debug("opt finished.")

            #################### testing for debug ############

            test_and_save['final'] = OptBEH.test_and_save('final', output_path, saver, sess, test, model)
            return test_and_save

    @staticmethod
    def train(model, sess, train_data, apply_grads):

        l = 0
        for d in train_data:
            feed_dict = model.beh_feed(d['action'], d['reward'], d['state'])
            loss, _ = sess.run([model.get_obj(),
                                apply_grads],
                               feed_dict=feed_dict)
            l += loss
        return l

    @staticmethod
    def test_and_save(label, output_path, saver, sess, test, model):
        output = [['sid', 'pre accu', 'post accu']]
        policies_list = {}
        state_track_list = {}
        if test is not None and len(test) > 0:
            DLogger.logger().debug("started testing...")
            for k in sorted(test.keys()):
                tr = test[k]
                policies_list[k] = {}
                post_accu = []
                state_track_list[k] = {}
                for v in tr:
                    policies, c_track, h_track, _ = model.simulate(sess,
                                                                   v['reward'],
                                                                   v['action'],
                                                                   v['state'])
                    policies_list[k][v['block']] = policies[0, :-1, :]
                    state_track_list[k][v['block']] = np.concatenate((c_track, h_track), axis=1)[0].T
                    post_accu.append(Assessor.evaluate_fit(policies[0, :-1, :], v['action'][0]))
                output.append([v['id'], None, np.array(post_accu)])
            DLogger.logger().debug("finished testing.")

            if output_path is not None:
                Export.write_to_csv(output, output_path, 'accuracy-' + str(label) + '.csv')
                Export.policies(policies_list, output_path, 'policies-' + str(label) + '.csv')
                Export.policies(state_track_list, output_path, 'state_track-' + str(label) + '.csv')

        #################### end of testing ############
        if output_path is not None and saver is not None:
            model_path = output_path + 'model-' + str(label) + '/'
            ensure_dir(model_path)
            saver.save(sess, model_path + 'model.cptk')
            train_writer = tf.summary.FileWriter(model_path)
            train_writer.add_graph(sess.graph)
            train_writer.flush()
            train_writer.close()

        return policies_list, state_track_list, output