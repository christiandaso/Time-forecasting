#Entrenamiento

import itertools
import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning, module="statsmodels")
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

#Optimizador
def tes_optimizer(train:pd.Series, test:pd.Series,
                  abg:list, 
                  step:int=48) -> (np.float64, np.float64, np.float64, np.float64, np.float64):
    best_alpha, best_beta, best_gamma, best_mae = None, None, None, float("inf")
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_trend=comb[1], smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        if mae < best_mae:
            best_alpha, best_beta, best_gamma, best_mae = comb[0], comb[1], comb[2], mae
        # print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2), round(mae, 2)])

    print("best_alpha:", round(best_alpha, 2), "best_beta:", round(best_beta, 2), "best_gamma:", round(best_gamma, 2),
          "best_mae:", round(best_mae, 4))

    print(y_pred)
    return best_alpha, best_beta, best_gamma, best_mae


#cargamos la tabla
def read_csvfile (filename):
    df = pd.read_csv(os.path.join('../data/processed/', filename))
    df.set_index('date', inplace = True)
    return df


def entrenamiento_modelo_optimizado (best_alpha, best_beta, best_gamma, best_mae, train):    
    #Optimizamos el modelo    
    model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12).\
            fit(smoothing_level=best_alpha, smoothing_trend=best_beta, smoothing_seasonal=best_gamma)
    
    #Guardamos el modelo entrenado
    package = '../models/best_model.pkl'
    pickle.dump(model, open(package, 'wb'))
    print("Modelo exportado correctamente")


#Entrenamiento completo
def main():
    train = read_csvfile('train_dataset.csv')
    valid = read_csvfile('valid_dataset.csv')
    print('Dataset cargados')

    alphas = betas = gammas = np.arange(0.20, 1, 0.10)
    abg = list(itertools.product(alphas, betas, gammas))

    best_alpha, best_beta, best_gamma, best_mae = tes_optimizer(train, valid, abg)

    entrenamiento_modelo_optimizado(best_alpha, best_beta, best_gamma, best_mae, train)
    print("Finalizó el entrenamiento del modelo")


if __name__ == "__main__":
    main()