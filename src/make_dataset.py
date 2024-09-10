import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
import os

# Definimos una ruta base para los datos
BASE_PATH = '../data/'

# Cargar la tabla
def read_raw_csv(filename, subfolder='raw'):
    filepath = os.path.join(BASE_PATH, subfolder, filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        print(f'{filename} cargado correctamente')
        return df
    else:
        print(f'Error: {filename} no encontrado en {subfolder}')
        return None

# Transformación del dataset
def transform_dataset(df):
    if df is not None:
        df['date'] = pd.to_datetime(df['date'], format="%d/%m/%Y")
        df.set_index('date', inplace=True)
        df = df[['co2']].resample('MS').mean()  # Resampleo mensual
        df.bfill(inplace=True)  # Relleno de valores nulos
        df.reset_index(inplace=True)
        return df
    else:
        print('DataFrame vacío o nulo, transformación no realizada')
        return None

# Guardado del CSV con los datos actualizados
def save_prepared_csv(filename, df, subfolder='processed'):
    if df is not None:
        filepath = os.path.join(BASE_PATH, subfolder, filename)
        df.to_csv(filepath, index=False)
        print(f'{filename} exportado correctamente en la carpeta {subfolder}')
    else:
        print(f'Error: No se pudo guardar {filename} debido a un DataFrame vacío o nulo')

# Matrices de entrenamiento/validación/scoring
def main():
    datasets = {
        'train': 'rawdata.csv',
        'valid': 'valid.csv',
        'scoring': 'scoring.csv'
    }

    for key, filename in datasets.items():
        df = read_raw_csv(filename)
        df = transform_dataset(df)
        save_prepared_csv(f'{key}_dataset.csv', df)

if __name__ == "__main__":
    main()
