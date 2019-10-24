# Extracting some info from subject data for on-policy simulations, such as number of trials etc.

from os.path import isdir
from actionflow.util import DLogger
import pandas as pd
from BD.data.data_reader import DataReader


def get_BD_confs():
    props = [[0.25, 0.05], [0.05, 0.25], [0.05, 0.125], [0.125, 0.05], [0.08, 0.05], [0.05, 0.08]]
    props += reversed(props)
    confs = []
    by_id = {}
    for e in DataReader.read_BD().groupby(['id', 'block']):
        trial = e[1]['block'].unique()
        block = {'block': trial[0],
                      'id': e[1]['id'].unique()[0],
                      'prop0': props[trial[0] - 1][0],
                      'prop1': props[trial[0] - 1][1],
                      'choices': e[1]['id'].size,
                      'group': e[1]['diag'].unique()[0]
                      }
        confs.append(block)

        if not block['id'] in by_id:
            by_id[block['id']] = [block]
        else:
            by_id[block['id']].append(block)

    by_group = {}
    subj_list = by_id.values()
    for s in subj_list:
        if not s[0]['group'] in by_group:
            by_group[s[0]['group']] = [s]
        else:
            by_group[s[0]['group']].append(s)

    return confs, by_id, by_group


def sims_analysis(input_folder, output_file, filter):
    all_train = merge_sim_files(filter, input_folder)
    chocie_stats = choice_statistics(all_train)
    chocie_stats.to_csv(output_file)


def merge_sim_files(filter, input_folder):
    from os import listdir
    from os.path import join
    sims_dirs = [f for f in listdir(input_folder) if isdir(join(input_folder, f))]
    total_sims = 0
    all_train = []
    for d in sims_dirs:
        DLogger.logger().debug(d)
        full_path = input_folder + d + '/'
        conf = pd.read_csv(full_path + 'config.csv')
        if filter(conf):
            train = pd.read_csv(full_path + 'train.csv')
            train['group'] = conf['group'][0]
            train['prop0'] = conf['prop0'][0]
            train['prop1'] = conf['prop1'][0]
            train['id'] = 'ID_' + conf['group'][0] + str(conf['N'][0])
            train['sim_ID'] = d
            all_train.append(train)
            total_sims += 1
    DLogger.logger().debug("total sims: " + str(total_sims))
    # concatinating all training inputs
    all_train = pd.concat(all_train, ignore_index=True)
    # finding wheher the best action was taken according to the planned probabilities
    best_action = 1 * (all_train['prop1'] > all_train['prop0'])
    all_train['best_action'] = 1 * (best_action == all_train['choices'])
    all_train['key'] = all_train['choices']
    return all_train


def choice_statistics(data):
    df = pd.DataFrame(columns=('id', 'group', 'best_sum',
                               'nonbest_sum',
                               'reward_stays', 'noreward_stays',
                               'reward_switches', 'noreward_switches',
                               ))

    ids = data['id'].unique().tolist()
    for id in ids:
        id_data = data.loc[data.id == id]
        sim_ids = id_data['sim_ID'].unique().tolist()
        best_sum = 0
        nonbest_sum = 0
        stays_reward = 0
        switches_reward = 0
        stays_noreward = 0
        switches_noreward = 0
        for sim_id in sim_ids:
            sim_data = id_data.loc[id_data.sim_ID == sim_id]
            next_choice = sim_data['choices'][1:].tolist() + [None]
            sim_data = sim_data.assign(next_choice=next_choice)
            sim_data = sim_data.assign(same=sim_data['choices'] == sim_data['next_choice'])
            best_sum += sum(sim_data['best_action'])
            nonbest_sum += sum(sim_data['best_action'] == 0)
            reward_trials = sim_data[:-1].loc[sim_data.reward == 1]
            noreward_trials = sim_data[:-1].loc[sim_data.reward == 0]
            stays_reward += sum(reward_trials['same'])
            stays_noreward += sum(noreward_trials['same'])
            switches_reward += sum(~reward_trials['same'])
            switches_noreward += sum(~noreward_trials['same'])
        df.loc[len(df)] = [id, id_data['group'].iloc[0], best_sum, nonbest_sum,
                           stays_reward,
                           stays_noreward,
                           switches_reward,
                           switches_noreward]

    return df


def extract_run_rew(data):
    ids = data['id'].unique().tolist()
    total_data = []
    for id in ids:
        DLogger.logger().debug('processing subject ' + id)
        id_data = data.loc[data.id == id]
        sim_ids = id_data['sim_ID'].unique().tolist()
        for sim_id in sim_ids:
            sim_data = id_data.loc[id_data.sim_ID == sim_id]

            next_key = sim_data['key'][1:].tolist() + [None]
            sim_data = sim_data.assign(next_key=next_key)
            sim_data = sim_data.assign(same=sim_data['key'] == sim_data['next_key'])
            sim_data.ix[sim_data.index[-1], 'same'] = 0

            last_key = None
            total_keys = 0
            total_rews = 0
            sim_data = sim_data.assign(prev_key=None)
            sim_data = sim_data.assign(prev_rewards=None)
            for i in range(sim_data.shape[0]):
                if last_key == sim_data.iloc[i]['key']:
                    sim_data.ix[sim_data.index[i], 'prev_key'] = total_keys
                    sim_data.ix[sim_data.index[i], 'prev_rewards'] = total_rews
                    total_keys += 1
                    total_rews += sim_data.iloc[i]['reward']
                else:
                    total_keys = 1
                    total_rews = sim_data.iloc[i]['reward']
                    sim_data.ix[sim_data.index[i], 'prev_key'] = 0
                    sim_data.ix[sim_data.index[i], 'prev_rewards'] = 0

                last_key = sim_data.iloc[i]['key']
            total_data.append(sim_data)
    all_trials = pd.concat(total_data)
    return all_trials
