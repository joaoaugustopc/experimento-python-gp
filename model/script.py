import runpy

funcao_train_valuen = input('Função para treino: ')
qtd_train_value = int(input('Quantidade de dados para treino: '))

best_score = float('-inf')
best_model = None
best_pred = None
best_iteration = None

for i in range(2):
    # Variáveis para passar para o módulo
    vars_to_pass = {'funcao_train': funcao_train_valuen, 'qtd_train': qtd_train_value, 'iteration': i}
    namespace = runpy.run_path('model/symb_reg.py', init_globals=vars_to_pass)
    
    score_gp = namespace['score_gp']
    est_gp = namespace['est_gp']
    y_pred = namespace['y_pred']

    if score_gp > best_score:
        best_score = score_gp
        best_model = est_gp
        best_pred = y_pred
        best_iteration = i
        
print('\nMelhor score ({best_iteration}):', best_score, '\nModelo:', best_model, '\nPrevisões:', best_pred)