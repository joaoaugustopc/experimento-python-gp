import pandas as pd

data = pd.read_csv('dataset/funcao11/funcao11_50.csv')
data_test = pd.read_csv('dataset/funcao11/funcao11_teste50k.csv')

x_train, y_train, val_esp_train = data[['x']], data['y'], data[['val_esp']]
x_test, y_test, val_esp_test = data_test[['x']], data_test['y'], data_test[['val_esp']]

from gplearn.genetic import SymbolicRegressor
import sympy as sp

#define operacoes

def protected_div(x1, x2):
    if x2 <= 1e-5:
        return 1
    return x1 / x2

def protected_sqrt(x1):
    if x1 < 0:
        return x1
    return sp.sqrt(x1)

def protected_log(x1):
    if x1 <= 0:
        return x1
    return sp.log(x1)

function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'cos', 'sen', 'tan', 'log', 'exp']

