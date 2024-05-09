import numpy as np
import pandas as pd

N_SAMPLES = 20_000 

def generate_5w_sentences(df, n_samples=N_SAMPLES, seed=2024):
    temp_d = {'who': df['who'].dropna(),
              'what': df['what'].dropna(),
              'when': df['when'].dropna(),
              'where': df['where'].dropna()}
    np.random.seed(2024)
    index = pd.MultiIndex.from_product(temp_d.values(), names=temp_d.keys())
    all_samples = pd.DataFrame(index=index).reset_index()
    training_set = all_samples.sample(n_samples).reset_index(drop=True).copy()
    training_set['sentences'] = training_set['who'] + ' ate ' + training_set['what'] + ' ' + training_set['when'] + ' ' + training_set['where'] + '.'
    return training_set

data_base = pd.read_csv('5W_dataset_base.csv')
data = generate_5w_sentences(data_base)
data.to_csv('5W_dataset.csv', index=False)