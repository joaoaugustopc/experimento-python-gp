import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

data = pd.read_csv(f'dataset/funcao{funcao_train}/funcao{funcao_train}_{qtd_train}.csv')
data_test = pd.read_csv(f'dataset/funcao{funcao_train}/funcao{funcao_train}_teste50k.csv')

#separando dados csv
X_train = data.drop('val-esp', axis=1).values
y_train = data['val-esp'].values

X_test = data_test.drop('val-esp', axis=1).values
y_test = data_test['val-esp'].values


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
        result = np.where(np.isnan(result), x1, result)
        result = np.where(np.isinf(result), x1, result)
    return result

def _protected_log(x1):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(x1 > 0, np.log(x1), x1)
        result = np.where(np.isnan(result), x1, result)
        result = np.where(np.isinf(result), x1, result)
    return result

protected_div = make_function(function=_protected_div, name='protected_div', arity=2)
protected_sqrt = make_function(function=_protected_sqrt, name='protected_sqrt', arity=1)
protected_log = make_function(function=_protected_log, name='protected_log', arity=1)

function_set = ['add', 'sub', 'mul', 'cos', 'sin', 'tan', protected_div, protected_sqrt, protected_log]

#cria o modelo: alterar parâmetros de mutação
est_gp = SymbolicRegressor(population_size=500, function_set=function_set, generations=50, tournament_size=4, metric='mse', p_crossover=0.9, init_depth=(5, 10), verbose=1, p_point_mutation=0.01, p_subtree_mutation=0.01, p_hoist_mutation=0.01, random_state=seed, parsimony_coefficient=3.0)

#treina o modelo
est_gp.fit(X_train, y_train)
# Previsões do modelo
y_pred = est_gp.predict(X_test)
#resultados em avaliação de modelo
apt = mean_squared_error(y_test, y_pred)

#imprime a função gerada e simplifica equação
import sympy as sp
import pandas as pd

converter = {
    'sub': lambda x, y : x-y,
    'protected_div': lambda x, y : x/y,
    'mul': lambda x, y : x*y,
    'add': lambda x, y : x+y,
    'pow': lambda x, y : x**y,
    'sin': lambda x: sp.sin(x),
    'cos': lambda x: sp.cos(x),
    'protected_sqrt': lambda x: sp.sqrt(x),
    'protected_log': lambda x: sp.log(x),
}

print("\nArvore de expressão:", est_gp._program)

exp_simp = sp.parse_expr(str(est_gp._program), converter)

print('\nEquação:', exp_simp)

df = pd.read_csv(f'results/2funcao{funcao_train}_{qtd_train}.csv')
df.loc[len(df.index)] = (seed, apt, exp_simp)
df.to_csv(f'results/2funcao{funcao_train}_{qtd_train}.csv', index=False)

"""
#plota a função
import matplotlib.pyplot as plt
plt.ion()

axs = plt.subplots(figsize=(12, 10))
#gráfico: Valores de teste vs Predições do modelo
axs[1].scatter(X_test[:, 0], y_test, color='green', alpha=0.5, label='Valores de teste')
axs[1].scatter(X_test[:, 0], y_pred, color='red', alpha=0.5, label='Predições do modelo')
axs[1].set_title(f'Valores de teste vs Predições do modelo: execução {iteration} - R2: {apt:.2f}')
axs[1].legend()
# Mostra os gráficos
plt.savefig(f'graficos/funcao{funcao_train}/data_{qtd_train}/valores_teste_vs_predicoes_execucao_{iteration}.png')
"""
