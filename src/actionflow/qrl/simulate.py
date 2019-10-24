import tensorflow as tf

from actionflow.data.data_process import DataProcess
from actionflow.qrl.opt_ml import OptML
from actionflow.util import DLogger
from actionflow.util.assessor import Assessor
import pandas as pd

from actionflow.util.export import Export
from actionflow.util.helper import format_to_training_data, Helper


class Simulator:
    def __init__(self):
        pass

    # @staticmethod
    # def simulate_worker(sess, worker, train, output_path):
    #
    #     policies = OptML.test_and_save("", output_path, None, sess, train, worker)
    #
    #     if output_path is not None:
    #         Export.export_train(train, output_path, 'train.csv')
    #
    #     return train, policies

    @staticmethod
    def simulate_env(sess, worker, output_path, n_trials, env_model, greedy=False):

        states, policies, rewards, choices = worker.simulate_env(sess, n_trials, env_model, greedy)
        train = format_to_training_data(rewards, choices, states)

        if output_path is not None:
            Export.export_train(train, output_path, 'train.csv')
            Export.policies({'id1': {'1': policies}}, output_path, 'policies-.csv')

        return train, policies

    @staticmethod
    def evaluate_CV(base_input_folder, base_output_folder, test_and_save, data, folds, model_iters, trials, random_tie=True):

        df = pd.DataFrame()

        saver = tf.train.Saver(max_to_keep=5)
        config = tf.ConfigProto(device_count={'GPU': 0})
        with tf.Session(config=config) as sess:
            for group in sorted(folds.keys()):
                    for fold in folds[group]:
                        for model_iter in model_iters:

                            input_folder = base_input_folder + '/' + group + '/' + fold + '/'

                            if base_output_folder is not None:
                                output_folder = base_output_folder + group + '/' + fold + '/' + model_iter + '/'
                            else:
                                output_folder = None

                            DLogger.logger().debug("input folder: {}".format(input_folder))
                            DLogger.logger().debug("output folder: {}".format(output_folder))

                            tr_tst = pd.read_csv(input_folder + 'train_test.csv')

                            if 'ID' in tr_tst:
                                DLogger.logger().debug("id column was not found. Replaced id column with ID.")
                                tr_tst['id'] = tr_tst['ID']

                            tst_ids = tr_tst.loc[tr_tst.train == 'test']['id']
                            dftr = pd.DataFrame({'id': tst_ids, 'train': 'train'})
                            train, _ = DataProcess.train_test_between_subject(data, dftr, trials)



                            model_path = input_folder + model_iter + '/'
                            ckpt = tf.train.get_checkpoint_state(model_path)
                            # tf.train.import_meta_graph(model_path + 'model.cptk.meta')
                            saver.restore(sess, ckpt.model_checkpoint_path)

                            policies = test_and_save(sess, train, output_folder)

                            if output_folder is not None:
                                Export.export_train(train, output_folder, 'train.csv')

                            train_merged = Export.merge_train(train)

                            #add a dummy column at the beginning
                            train_merged.insert(loc=0, column='tmp', value='')

                            policies_merged = Export.merge_policies(policies)

                            #add a dummy column at the beginning
                            policies_merged.insert(loc=0, column='tmp', value='')

                            # train_merged = pd.read_csv(output_folder + 'train.csv')
                            # policies_merged = pd.read_csv(output_folder + 'policies.csv')

                            acc, nlp, total_nlp = Assessor.evaluate_fit_multi(policies_merged,
                                                                   train_merged,
                                                                   pol_in_log=True,
                                                                   random_tie=random_tie
                                                                   )
                            df = df.append(pd.DataFrame({
                                'acc': [acc],
                                'nlp': [nlp],
                                'total nlp': [total_nlp],
                                'group': group,
                                'option': Helper.dicstr({}),
                                'fold': fold,
                                'model_iter': model_iter
                            }))

        if base_output_folder is not None:
            df.to_csv(base_output_folder + 'accu.csv')

        return df