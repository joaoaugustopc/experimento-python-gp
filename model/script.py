import runpy
import os
import random 
import pandas as pd

funcao_train_valuen = input('Função para treino: ')
qtd_train_value = int(input('Quantidade de dados para treino: '))

# Criar uma nova pasta chamada "graficos/funcao"
os.makedirs(f'graficos/funcao{funcao_train_valuen}/data_{qtd_train_value}', exist_ok=True)

best_score = float('-inf')
best_model = None
best_pred = None
best_iteration = None
y_pred = []
mse_list = []

for i in range(30):
    # Variáveis para passar para o módulo
    vars_to_pass = {'funcao_train': funcao_train_valuen, 'qtd_train': qtd_train_value, 'iteration': i, 'seed' : random.randint(0, 1000)}
    namespace = runpy.run_path('model/symb_reg.py', init_globals=vars_to_pass)
    
    y_pred.append(namespace['y_pred'])
    mse_list.append(namespace['apt'])
    
    score_gp = namespace['score_gp']
    est_gp = namespace['est_gp']

    if score_gp > best_score:
        best_score = score_gp
        best_model = est_gp
        best_pred = y_pred
        best_iteration = i
        

dados_mse = pd.DataFrame(mse_list, columns=['MSE'])
mediana = dados_mse['MSE'].median()
iqr = dados_mse['MSE'].quantile(0.75) - dados_mse['MSE'].quantile(0.25)

print('Best model:', best_model)
print('Dados:\nMediana:', mediana, '\nIQR:', iqr)