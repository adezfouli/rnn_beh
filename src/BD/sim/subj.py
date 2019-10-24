# Analysis of the subjects data.

from BD.data.data_reader import DataReader
from BD.sim.sims import choice_statistics, extract_run_rew
from BD.util.paths import Paths


def analysis_BD():

    output_file = Paths.local_path + 'BD/to_graph_data/subj_stats.csv'

    data = DataReader.read_BD()
    data['best_action'] = 1 * (data['choice'] == 'R1')
    data['choices'] = data['choice']
    data['sim_ID'] = data['id'] + str(data['block'])
    data['group'] = data['diag']

    chocie_stats = choice_statistics(data)
    chocie_stats.to_csv(output_file)


def analysis_action_reward():

    data = DataReader.read_BD()
    data['best_action'] = 1 * (data['choice'] == 'R1')
    data['sim_ID'] = (data['id'] + data['block'].map(str))
    data['group'] = data['diag']

    all_trials = extract_run_rew(data)

    # reward_trials = pd.concat(total_data)
    # reward_trials = reward_trials.loc[reward_trials.reward == 1]
    # reward_trials.loc[reward_trials['prev_rewards'] == 0, 'pre_rew_group'] = "0"
    # reward_trials.loc[reward_trials['prev_rewards'] == 1, 'pre_rew_group'] = "1"
    # reward_trials.loc[reward_trials['prev_rewards'] == 2, 'pre_rew_group'] = "2"
    # reward_trials.loc[reward_trials['prev_rewards'] > 2, 'pre_rew_group'] = ">2"
    #
    # reward_trials.loc[reward_trials['prev_key'] < 5, 'prev_key_group'] = "<10"
    # reward_trials.loc[(reward_trials['prev_key'] >= 5) & (reward_trials['prev_key'] < 10), 'prev_key_group'] = "10-20"
    # reward_trials.loc[(reward_trials['prev_key'] >= 10) & (reward_trials['prev_key'] < 15), 'prev_key_group'] = ">=10"
    #
    # grouped = reward_trials.groupby(['prev_key_group', 'pre_rew_group', 'group'])['same'].mean()
    # output_file = Paths.local_path + 'to_graph_data/subj_stats_pre_rew_key.csv'
    # grouped.to_csv(output_file, header=True)
    #
    # grouped = reward_trials.groupby(['prev_key_group', 'pre_rew_group', 'ID', 'group'])['same'].mean()
    # output_file = Paths.local_path + 'to_graph_data/subj_stats_pre_rew_key_ID.csv'
    # grouped.to_csv(output_file, header=True)
    #
    # grouped = reward_trials.groupby(['prev_key_group', 'ID', 'group'])['same'].mean()
    # output_file = Paths.local_path + 'to_graph_data/subj_stats_pre_key_ID.csv'
    # grouped.to_csv(output_file, header=True)
    #
    # grouped = reward_trials.groupby(['pre_rew_group', 'ID', 'group'])['same'].mean()
    # output_file = Paths.local_path + 'to_graph_data/subj_stats_pre_rew_ID.csv'
    # grouped.to_csv(output_file, header=True)
    #
    # grouped = reward_trials.groupby(['pre_rew_group', 'prev_key_group', 'ID', 'group'])['prev_key_group'].count()
    # output_file = Paths.local_path + 'to_graph_data/subj_run_length_ID.csv'
    # grouped.to_csv(output_file, header=True)

    output_file = Paths.local_path + 'BD/to_graph_data/subj_all_data.csv'
    all_trials.to_csv(output_file, header=True)

    # no_reward_trials = all_trials.loc[no_reward_trials.prev_rewards == 0]
    # grouped = no_reward_trials.groupby(['prev_key', 'ID', 'group'])['same'].mean()
    # output_file = Paths.local_path + 'to_graph_data/subj_stats_pre_key_count.csv'
    # grouped.to_csv(output_file, header=True)


if __name__ == '__main__':
    analysis_BD()
    analysis_action_reward()
