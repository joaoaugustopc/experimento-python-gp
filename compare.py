
import pandas as pd

funcao_train_valuen = input('Função para treino: ')
qtd_train_value = int(input('Quantidade de dados para treino: '))

data = pd.read_csv(f'results/2funcao{funcao_train_valuen}_{qtd_train_value}.csv')
mediana = data['mse'].median()
iqr = data['mse'].quantile(0.75) - data['mse'].quantile(0.25)

print('Dados:\nMediana:', mediana, '\nIQR:', iqr)

data = pd.read_csv(f'results/sggp_funcao{funcao_train_valuen}_{qtd_train_value}.csv')
mediana = data['mse'].median()
iqr = data['mse'].quantile(0.75) - data['mse'].quantile(0.25)

print('Dados:\nMediana:', mediana, '\nIQR:', iqr)