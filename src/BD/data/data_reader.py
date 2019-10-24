import pandas as pd

class DataReader:
    def __init__(self):
        pass

    @staticmethod
    def read_BD():
        data = pd.read_csv("../data/BD/choices_diagno.csv.zip", compression='zip', header=0, sep=',', quotechar='"')
        data['reward'] = [0 if x == 'null' else 1 for x in data['outcome']]
        data['id'] = data['ID']
        data['block'] = data['trial']

        #R1: right, R2: Left
        data['action'] = [0 if x == 'R1' else 1 for x in data['key']]
        del data['trial']
        del data['ID']
        return data

    @staticmethod
    def read_BD_index():
        return pd.read_csv("../data/BD/ind.csv", header=0, sep=',', quotechar='"')

if __name__ == '__main__':
    DataReader.read_BD()
