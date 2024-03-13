import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

data_teste = pd.read_csv('dataset/funcao13/funcao13_teste50k.csv')
val_esp = data_teste['val-esp'].values
data = pd.read_csv('results/funcao13_50.csv')
equacoes = data['equacao'].values

df = pd.DataFrame(columns=['equacao','mse'])

X0 = data_teste['a'].values
X1 = data_teste['b'].values
X2 = data_teste['c'].values
X3 = data_teste['d'].values
X4 = data_teste['e'].values



for i in range(len(equacoes)):
    equacoes[i]=str(equacoes[i])
    if equacoes[i] == 'nan':
        result = 0
    else:
        result = eval(equacoes[i])
    if np.isscalar(result):
        result = np.full(50000,result)
    apt = mean_squared_error(val_esp, result)
    df.loc[len(df.index)] = (equacoes[i],apt)


df.to_csv('results/resultados_equacoes_funcao13_50.csv', index=False)