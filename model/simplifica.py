import pandas as pd
import sympy as sp
import numpy as np

from sympy import Piecewise

file_trees = open('tests/trees_simp.txt', 'r')

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
    'tan': lambda x: sp.tan(x),
}

exp_simp = sp.parse_expr(str(file_trees.read()), converter)

print("conversao:", exp_simp)

simplif_trig = sp.trigsimp(exp_simp)

simplif = sp.simplify(simplif_trig)

print("\nsimplificacao:", simplif)
