from SGGP.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


data_train = pd.read_csv(f'dataset/funcao{funcao_train}/funcao{funcao_train}_{qtd_train}.csv')
data_test = pd.read_csv(f'dataset/funcao{funcao_train}/funcao{funcao_train}_teste50k.csv')

X_train = data_train.drop(["val-esp"], axis=1).values
y_train = data_train['val-esp'].values

X_test = data_test.drop(["val-esp"], axis=1).values
y_test = data_test['val-esp'].values


est_gp = SymbolicRegressor(population_size=500,generations=50, stopping_criteria=1e-5,
                    verbose=1, random_state=seed,n_features=X_train.shape[1], metric='mse', 
                    tournament_size= 4, init_depth=5)

est_gp.fit(X_train, y_train)

y_predict = est_gp.predict(X_test)[0]

apt = mean_squared_error(y_test, y_predict)

df = pd.read_csv(f'results/sggp_funcao{funcao_train}_{qtd_train}.csv')
df.loc[len(df.index)] = (seed, apt, est_gp._program)
df.to_csv(f'results/sggp_funcao{funcao_train}_{qtd_train}.csv', index=False)


""" 
result = pd.DataFrame({'Id': data_test['remainder__Id'], 'SalePrice': y_predict})

result['Id'] = result['Id'].astype(int)
result.to_csv('submissions/sample_submission_sggp.csv', index=False)
"""

