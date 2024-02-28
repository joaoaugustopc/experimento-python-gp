import pandas as pd
import numpy as np

data = pd.read_csv('dataset/funcao11/funcao11_50.csv')
data_test = pd.read_csv('dataset/funcao11/funcao11_teste50k.csv')


X_train = np.column_stack((data['x'], data['y']))
y_train = data['val-esp']

X_test = np.column_stack((data_test['x'], data_test['y']))
y_test = data_test['val-esp']

from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn import set_config
set_config(transform_output = "pandas")
import sympy as sp
import numpy as np

#define operacoes protegidas
def _protected_div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(x2) > 0.00001, x1 / x2, 1.)
        result = np.where(np.isnan(result), 1., result)
        result = np.where(np.isinf(result), 1., result)
    return result

def _protected_sqrt(x1):
    with np.errstate(invalid='ignore'):
        result = np.where(x1 >= 0, np.sqrt(x1), x1)
        result = np.where(np.isnan(result), 0., result)
        result = np.where(np.isinf(result), 0., result)
    return result

def _protected_log(x1):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(x1 > 0, np.log(x1), x1)
        result = np.where(np.isnan(result), 0., result)
        result = np.where(np.isinf(result), 0., result)
    return result

protected_div = make_function(function=_protected_div, name='protected_div', arity=2)
protected_sqrt = make_function(function=_protected_sqrt, name='protected_sqrt', arity=1)
protected_log = make_function(function=_protected_log, name='protected_log', arity=1)

function_set = ['add', 'sub', 'mul', 'cos', 'sin', 'tan', protected_div, protected_sqrt, protected_log]

converter ={
    'add': '+',
    'sub': '-',
    'mul': '*',
    'cos': 'cos',
    'sin': 'sin',
    'tan': 'tan',
    'protected_div': '/',
    'protected_sqrt': 'sqrt',
    'protected_log': 'log',
}

#cria o modelo: alterar parâmetros de mutação
est_gp = SymbolicRegressor(population_size=500, function_set=function_set, generations=50, tournament_size=4, metric='mse', p_crossover=0.9, init_depth=(5, 10), verbose=1, p_point_mutation=0.1, p_subtree_mutation=0,p_hoist_mutation=0)

#treina o modelo
est_gp.fit(X_train, y_train)

#resultados em avaliação de modelo
print(est_gp.score(X_test, y_test))

"""
#simplificando expressões
simp_exp = sp.sympify(str(est_gp._program), locals=converter)
print(simp_exp)
"""
    
