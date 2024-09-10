#Scoring

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


#Cargamos la tabla
def score_model (filename, scores):
    df = pd.read_csv(os.path.join('../data/processed/', filename))
    df.set_index('date', inplace = True)
    print(filename, 'cargado correctamente')
    print(df)
    
    #Lectura del modelo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package,'rb'))
    print("Modelo importado")

    #Predecimos sobre el df scoring
    df['co2'] = model.predict(start=df.index.min(), end=df.index.max())
    df.rename(columns = {'co2':'co2_predicted'}, inplace = True)
    df = df.reset_index()
    df.to_csv(os.path.join('../data/scores/', scores), index = False)
    print(scores, 'exportado correctamente')


#Entrenamiento completo
def main():
    score_model('scoring_dataset.csv', 'predicted_n3moths.csv')
    print("Finalizó la predicción del modelo")


if __name__ == "__main__":
    main()