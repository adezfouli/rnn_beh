import h5py

from actionflow.rnn.lstm_beh import LSTMBeh
from actionflow.util import DLogger
from actionflow.util.helper import format_to_training_data
from ..data.data_process import DataProcess
from ..rnn.opt_beh import OptBEH
from ..util.assessor import Assessor
from ..util.export import Export
import tensorflow as tf
import pandas as pd


class Simulator:
    def __init__(self):
        pass

    @staticmethod
    def simulate_worker(worker, model_path, train, output_path):

        saver = tf.train.Saver(max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state(model_path)
        config = tf.ConfigProto(device_count={'GPU': 0})

        with tf.Session(config=config) as sess:
            saver.restore(sess, ckpt.model_checkpoint_path)

            policies_list, state_track_list, output = OptBEH.test_and_save("", output_path, None, sess, train, worker)

            if output_path is not None:
                Export.export_train(train, output_path, 'train.csv')


        return policies_list, state_track_list, output




    @staticmethod
    def simulate_env(worker, model_path, output_path, n_trials, env_model,
                     greedy=False, export_state_track=True, cell_mask=None):

        import numpy as np
        saver = tf.train.Saver(max_to_keep=5)
        ckpt = tf.train.get_checkpoint_state(model_path)
        config = tf.ConfigProto(device_count={'GPU': 0})

        with tf.Session(config=config) as sess:
            tf.train.import_meta_graph(model_path + 'model.cptk.meta')
            saver.restore(sess, ckpt.model_checkpoint_path)
            states, policies, rewards, choices, rnn_states = worker.simulate_env(sess,
                                                                                 n_trials,
                                                                                 env_model,
                                                                                 greedy,
                                                                                 cell_mask=cell_mask)

            train = format_to_training_data(rewards, choices, states)

            if output_path is not None:
                Export.policies({'id1': {'1': policies}}, output_path, 'policies-.csv')
                if export_state_track:
                    Export.policies({'id1': {'1': np.concatenate(rnn_states, axis=1)}}, output_path, 'state_track-.csv')
                Export.export_train(train, output_path, 'train.csv')

        return train, policies, rnn_states


    @staticmethod
    def evaluate_CV(worker_gen, base_input_folder, base_output_folder,
                    cells,
                    n_actions,
                    n_states,
                    data, folds, model_iters, trials):
        df = pd.DataFrame()
        for group in folds.keys():
            for n_cells in cells:
                for fold in folds[group]:
                    for model_iter in model_iters:
                        input_folder = base_input_folder + str(n_cells) + 'cells/' + group + '/' + fold + '/'
                        output_folder = base_output_folder + str(
                            n_cells) + 'cells/' + group + '/' + fold + '/' + model_iter + '/'

                        tr_tst = pd.read_csv(input_folder + 'train_test.csv')

                        if not ('id' in tr_tst):
                            DLogger.logger().debug('id not found in test train file. Using ID instead.')
                            tr_tst['id'] = tr_tst['ID']

                        tst_ids = tr_tst.loc[tr_tst.train == 'test']['id']
                        dftr = pd.DataFrame({'id': tst_ids, 'train': 'train'})
                        train, _ = DataProcess.train_test_between_subject(data, dftr, trials)

                        tf.reset_default_graph()

                        worker = worker_gen(n_actions, n_states, n_cells)
                        DLogger.logger().debug(input_folder + model_iter)

                        Simulator.simulate_worker(worker,
                                                  input_folder + model_iter + '/',
                                                  train,
                                                  output_folder)

                        train = pd.read_csv(output_folder + 'train.csv')
                        policies = pd.read_csv(output_folder + 'policies-.csv')

                        acc, nlp, total_nlp = Assessor.evaluate_fit_multi(policies, train)
                        df = df.append(pd.DataFrame({
                            'acc': [acc],
                            'nlp': [nlp],
                            'total nlp': [total_nlp],
                            'group': group,
                            'cell': n_cells,
                            'fold': fold,
                            'model_iter': model_iter
                        }))
        df.to_csv(base_output_folder + 'accu.csv')
