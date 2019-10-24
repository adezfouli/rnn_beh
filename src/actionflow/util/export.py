import csv
import os
import pandas as pd


class Export:

    @staticmethod
    def write_to_csv(data, path, output_file):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + output_file, "w") as f:
            writer = csv.writer(f)
            writer.writerows(data)


    @staticmethod
    def policies(policies, path, file_name):
        df_all = Export.merge_policies(policies)

        if not os.path.exists(path):
            os.makedirs(path)

        df_all.to_csv(path + file_name)

    @staticmethod
    def merge_policies(policies):
        dfs = []
        for id, pol in policies.items():
            for t, v in pol.items():
                df = pd.DataFrame(v)
                df['id'] = id
                df['block'] = t
                dfs.append(df)
        df_all = pd.concat(dfs)
        return df_all

    @staticmethod
    def export_train(train, path, file_name):
        df_all = Export.merge_train(train)

        if not os.path.exists(path):
            os.makedirs(path)

        df_all.to_csv(path + file_name)

    @staticmethod
    def merge_train(train):
        dfs = []
        for id in sorted(train.keys()):
            tr = train[id]
            for t in range(len(tr)):
                df = pd.DataFrame()
                df['reward'] = tr[t]['reward'][0]
                df['action'] = tr[t]['action'][0]
                if tr[t]['state'] is not None:
                    for s in range(tr[t]['state'][0].shape[1]):
                        df['state' + str(s)] = tr[t]['state'][0][:, s]
                df['id'] = id
                df['block'] = t + 1
                dfs.append(df)
        df_all = pd.concat(dfs)
        return df_all
