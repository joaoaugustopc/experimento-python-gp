import runpy
import os
import random 
import pandas as pd

funcao_train_valuen = input('Função para treino: ')
qtd_train_value = int(input('Quantidade de dados para treino: '))

""" 
# Criar uma nova pasta chamada "graficos/funcao"
os.makedirs(f'graficos/funcao{funcao_train_valuen}/data_{qtd_train_value}', exist_ok=True)

# Criar um novo arquivo chamado "results/funcao.csv" para salvar os resultados
df = pd.DataFrame(columns=['seed', 'mse', 'equacao'])
df.to_csv(f'results/funcao{funcao_train_valuen}_{qtd_train_value}.csv', index=False)

"""

for i in range(1):
    # variáveis para passar para o módulo
    vars_to_pass = {'funcao_train': funcao_train_valuen, 'qtd_train': qtd_train_value, 'iteration': i, 'seed' : random.randint(0, 1000)}
    namespace = runpy.run_path('model/symb_reg.py', init_globals=vars_to_pass)
        
data = pd.read_csv(f'results/funcao{funcao_train_valuen}_{qtd_train_value}.csv')
mediana = data['mse'].median()
iqr = data['mse'].quantile(0.75) - data['mse'].quantile(0.25)

print('Dados:\nMediana:', mediana, '\nIQR:', iqr)
