import pandas as pd
import numpy as np
import random

path = "data/Estandarizado.csv"
path_out = f"output/Q/{random.randint(0,1000)}_preprocessed_data.csv"

def import_data(path):
    df = pd.read_csv(path, sep=";", decimal=",")
    return df

def preprocess_data(dfa, max_outliers=1):  
    non_numeric_cols = dfa.select_dtypes(exclude=[np.number])
    df = dfa.select_dtypes(include=[np.number])
    
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    
    buttom_limit = Q1 - 3 * IQR
    top_limit = Q3 + 3 * IQR
    
    outlier_count = ((df < buttom_limit) | (df > top_limit)).sum(axis=1)
    
    parametro = outlier_count > max_outliers
    
    df_arreglado = df[~parametro]
    non_numeric_filtered = non_numeric_cols[~parametro]
    df_final = pd.concat([non_numeric_filtered, df_arreglado], axis=1)
    
    return df_final

def export_data(df, path):
    df.to_csv(path, index=False, sep=";", decimal=".")
    


df = import_data(path)

df_procesado = preprocess_data(df)

export_data(df_procesado, path_out)
print(path_out)
print(len(df_procesado))
