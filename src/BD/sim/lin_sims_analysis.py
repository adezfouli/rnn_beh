# Analysis the data generated from on policy simulations of LIN.

from actionflow.util.helper import ensure_dir
from BD.sim.sims import sims_analysis, merge_sim_files, extract_run_rew
from BD.util.paths import Paths


def sims_analysis_BD():
    input_folder = Paths.rest_path + 'archive/beh/lin-on-sims/'
    output_folder = Paths.local_path + 'BD/to_graph_data/'
    ensure_dir(output_folder)
    sims_analysis(input_folder,
                  output_folder + 'lin_onpolicy.csv',
                  lambda x: True
                  )


# run this after the previous line finished:
if __name__ == '__main__':
    sims_analysis_BD()

    data = merge_sim_files(lambda x: True, Paths.rest_path + 'archive/beh/lin-on-sims/')
    all_trials = extract_run_rew(data)

    output_file = Paths.local_path + 'BD/to_graph_data/lin_all_data.csv'
    all_trials.to_csv(output_file, header=True)
