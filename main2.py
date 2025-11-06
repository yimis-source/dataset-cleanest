import pandas as pd
import numpy as np
import random

path = "data/Estandarizado.csv"
path_out = f"output/Z/{random.randint(0,1000)}_preprocessed_data.csv"

def import_data(path):
    df = pd.read_csv(path, sep=";", decimal=",")
    return df

def preprocess_data_zscore(dfa, threshold=3):
    non_numeric = dfa.select_dtypes(exclude=[np.number])
    df = dfa.select_dtypes(include=[np.number])

    mean = df.mean()
    std = df.std(ddof=0)

    z_normalized = (df - mean) / std

    outliers = (np.abs(z_normalized) > threshold).any(axis=1)

    df_arreglado = df[~outliers]
    non_numeric_filtered = non_numeric[~outliers]

    df_final = pd.concat([non_numeric_filtered, df_arreglado], axis=1)

    return df_final

def export_data(df, path):
    df.to_csv(path, index=False, sep=";", decimal=".")



df = import_data(path)
df_procesado = preprocess_data_zscore(df, threshold=3)
export_data(df_procesado, path_out)
print(path_out)
print(len(df_procesado))

