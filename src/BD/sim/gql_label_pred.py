# Making prediction about diagnostic labels of the subjects. Note that this file needs
# the output of 'fit/gql_ml_pred.py'.

from BD.sim.rnn_label_pred import finding_CV
from actionflow.data.data_process import DataProcess
from actionflow.qrl.gql import GQL
from actionflow.qrl.opt_ml import OptML
from actionflow.util import DLogger
from actionflow.util.helper import load_model
from BD.data.data_reader import DataReader
from BD.util.paths import Paths
import tensorflow as tf
import pandas as pd


def GQL_classify_subjects():
    tf.reset_default_graph()

    data = DataReader.read_BD()
    ids = data['id'].unique().tolist()
    dftr = pd.DataFrame({'id': ids, 'train': 'train'})
    train, test = DataProcess.train_test_between_subject(data, dftr,
                                                         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    model_iter = 'model-final'

    df = pd.DataFrame(columns=('model', 'id', 'loss'))
    config = tf.ConfigProto(device_count={'GPU': 0})

    subj_paths = finding_CV(Paths.rest_path + 'archive/beh/gql-ml-pred-diag/')

    worker = GQL.get_instance(2, 2, {})
    worker.set_params(OptML.get_variables(worker.get_params()))

    for k, tr in train.iteritems():
        for g, p in subj_paths[k].iteritems():

            DLogger.logger().debug('subject ' + k + ' group ' + g + ' path ' + p)
            model_path = p + model_iter + '/'
            with tf.Session(config=config) as sess:
                load_model(sess, model_path)
                total_loss = 0
                for v in tr:
                    ell, _, _ = worker.simulate(sess, v['reward'], v['action'])
                    total_loss += -ell

                df.loc[len(df)] = [g, k, total_loss]

    df.to_csv(Paths.local_path + 'BD/gql_diag.csv')


if __name__ == '__main__':
    GQL_classify_subjects()
