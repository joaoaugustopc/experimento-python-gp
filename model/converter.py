import pandas as pd
import numpy as np

# Load the data

data = pd.read_csv('results_c++/avaliacaoFuncao5_arvores50.csv',header=None)


for i in range(len(data)):
    
    arv = data.loc[i].values

    # stack = ['6','x','$','y','$','*']

    stack = list(arv)

    func_set = {
        '+': 'add',
        '-': 'sub',
        '*': 'mul',
        '/': 'protected_div',
        '^': 'pow',
        '$': 'sin',
        '#': 'protected_sqrt',
        '!': 'protected_log',
        '&': 'tanh',
        'e': 'exp'
    }

    def remove_last_element(lst):
        if lst:  # check if the list is not empty
            lst.pop()
        return lst

    def converter(stack):
        if not stack:
            return ""
        if stack[-1] in func_set:
            if(stack[-1] == '$' or stack[-1] == '#' or stack[-1] == '!' or stack[-1] == '&' or stack[-1] == 'e'):
                opr = stack.pop()
                val1 = converter(stack)
                return f"{func_set[opr]}({val1})"
            else:
                opr = stack.pop()
                val1 = converter(stack)
                val2 = converter(stack)
                return f"{func_set[opr]}({val2},{val1})"
        else:
            return stack.pop()
        

    equacao = converter(stack)
    print(equacao)

