#Validacion

import pandas as pd
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.tsa.api as smt
import pickle
import os


#cargamos la tabla
def eval_model (filename):
    df = pd.read_csv(os.path.join('../data/processed/', filename))
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    df.set_index('date', inplace=True)
    print(filename, 'cargado correctamente')

    #Lectura del modelo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package,'rb'))
    print("Modelo importado")

    #Predicción
    y_pred = model.forecast(48)
    mae = mean_absolute_error(df, y_pred)
    print("MAE: ", mae)


#Entrenamiento completo
def main():
    eval_model('valid_dataset.csv')
    print("Finalizó la validación del modelo")


if __name__ == "__main__":
    main()