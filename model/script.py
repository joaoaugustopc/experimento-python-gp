import runpy
import os
import random 
import pandas as pd

funcao_train_valuen = input('Função para treino: ')
qtd_train_value = int(input('Quantidade de dados para treino: '))

# Criar um novo arquivo chamado "results/funcao.csv" para salvar os resultados
file_path = f'results/2funcao{funcao_train_valuen}_{qtd_train_value}.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
# Criar um novo arquivo chamado "results/funcao.csv" para salvar os resultado
    df = pd.DataFrame(columns=['seed', 'mse', 'equacao'])
    df.to_csv(f'results/2funcao{funcao_train_valuen}_{qtd_train_value}.csv', index=False)


file_path = f'results/sggp_funcao{funcao_train_valuen}_{qtd_train_value}.csv'
if os.path.exists(file_path):
    df2 = pd.read_csv(file_path)
else:
# Criar um novo arquivo chamado "results/funcao.csv" para salvar os resultado
    df2 = pd.DataFrame(columns=['seed', 'mse', 'equacao'])
    df2.to_csv(f'results/sggp_funcao{funcao_train_valuen}_{qtd_train_value}.csv', index=False)

for i in range(20):
    # variáveis para passar para o módulo
    vars_to_pass = {'funcao_train': funcao_train_valuen, 'qtd_train': qtd_train_value, 'iteration': i, 'seed' : random.randint(0, 1000)}
    namespace = runpy.run_path('model/symb_reg.py', init_globals=vars_to_pass)
    namespace = runpy.run_path('model/sggp.py', init_globals=vars_to_pass)
        

