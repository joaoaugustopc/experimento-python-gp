import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

data_teste = pd.read_csv('dataset/funcao5/funcao5_teste50k.csv')
val_esp = data_teste['val-esp'].values
data = pd.read_csv('results/funcao5_50.csv')
equacoes = data['equacao'].values

df = pd.DataFrame(columns=['equacao','mse'])

X0 = data_teste['X1'].values
X1 = data_teste['X2'].values

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


df.to_csv('results/resultados_equacoes_funcao5_50.csv', index=False)