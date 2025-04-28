import pandas as pd
def load_data():
    dados = pd.read_csv('train.csv')
    return dados