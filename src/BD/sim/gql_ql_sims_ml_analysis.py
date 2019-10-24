# Analysis the data generated from on policy simulations of QL, QLP and GQL.

from BD.sim.sims import sims_analysis, merge_sim_files, extract_run_rew
from BD.util.paths import Paths


def sims_analysis_BD():
    input_folder = Paths.rest_path + 'archive/beh/qlp-ml-opt/qlp-ml/'
    sims_analysis(input_folder,
                  Paths.local_path + 'BD/to_graph_data/qlp_ml_onpolicy__stats.csv',
                  lambda conf: True
                  )

    input_folder = Paths.rest_path + 'archive/beh/ql-ml-opt/ql-ml/'
    sims_analysis(input_folder,
                  Paths.local_path + 'BD/to_graph_data/ql_ml_onpolicy_stats.csv',
                  lambda conf: True
                  )


def sims_analysis_GQL_BD():
    input_folder = Paths.rest_path + 'archive/beh/gql-ml-opt/gql-ml/'

    sims_analysis(input_folder,
                  Paths.local_path + 'BD/to_graph_data/gql_ml_onpolicy_stats.csv',
                  lambda conf: True
                  )

    input_folder = Paths.rest_path + 'archive/beh/gql10d-ml-opt/gql10d-ml/'

    sims_analysis(input_folder,
                  Paths.local_path + 'BD/to_graph_data/gql10d_ml_onpolicy_stats.csv',
                  lambda conf: True
                  )


if __name__ == '__main__':
    sims_analysis_BD()
    sims_analysis_GQL_BD()

    data = merge_sim_files(lambda x: True, Paths.rest_path + 'archive/beh/gql10d-ml-opt/gql10d-ml/')
    all_trials = extract_run_rew(data)

    output_file = Paths.local_path + 'BD/to_graph_data/gql10d_all_data_ml.csv'
    all_trials.to_csv(output_file, header=True)

    data = merge_sim_files(lambda x: True, Paths.rest_path + 'archive/beh/gql-ml-opt/gql-ml/')
    all_trials = extract_run_rew(data)

    output_file = Paths.local_path + 'BD/to_graph_data/gql_all_data_ml.csv'
    all_trials.to_csv(output_file, header=True)
