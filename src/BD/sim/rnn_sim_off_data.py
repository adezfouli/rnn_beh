import pandas as pd
from actionflow.data.data_process import DataProcess
from actionflow.rnn.lstm_beh import LSTMBeh
from actionflow.rnn.simulate import Simulator
from actionflow.util import DLogger
from BD.data.data_reader import DataReader
import tensorflow as tf
from BD.util.paths import Paths


def simulate_model(input_folder, output_folder, data, n_cells):

    dftr = pd.DataFrame({'id': data['id'].unique().tolist(), 'train': 'train'})
    train, _ = DataProcess.train_test_between_subject(data, dftr,
                                                      [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    tf.reset_default_graph()

    worker = LSTMBeh(2, 0, n_cells)
    DLogger.logger().debug('started simulations')

    Simulator.simulate_worker(worker,
                              input_folder,
                              train,
                              output_folder)


if __name__ == '__main__':
    data = DataReader.read_BD()

    simulate_model(Paths.rest_path + 'archive/beh/rnn-opt-from-init/10cells/Healthy/model-final/',
                   Paths.local_path + 'BD/on-sims-data/Healthy/',
                   data.loc[data.diag == 'Healthy'],
                   10
                   )


    simulate_model(Paths.rest_path + 'archive/beh/rnn-opt-from-init/10cells/Depression/model-final',
                   Paths.local_path + 'BD/on-sims-data/Depression/',
                   data.loc[data.diag == 'Depression'],
                   10
                   )

    simulate_model(Paths.rest_path + 'archive/beh/rnn-opt-from-init/20cells/Bipolar/model-final',
                   Paths.local_path + 'BD/on-sims-data/Bipolar/',
                   data.loc[data.diag == 'Bipolar'],
                   20
                   )
