# -*- coding: utf-8 -*-

import sys
sys.path.append("../")
sys.path.append("../Tests_functions")

from time import time
import matplotlib.pyplot as plt
import numpy as np
import random
from test_functions import sinuses
from test_functions import quadratic_function as qf
from method_functions import main_solver
import math

def test_research_solve(epsilon):
	num = 0
	N = 10
	results = []
	for eps in epsilon:
		results = []
		for i in range(N):
			param = np.random.uniform(-10, 10, 6)
			param[2] =  abs(param[2])
			f = qf(param)
			size_1, size_2 = random.uniform(0.5, 1), random.uniform(0.5, 1)
			x_1, y_1 = f.solution[0], f.solution[1]
			Q = [x_1 - (1-size_1), x_1 + (1+size_1), y_1 - (1-size_2), y_1 + (1+size_2)]
			
			m1 = time()
			solver = main_solver(f, Q, eps)
			solver.init_help_function()
			res_1 = solver.halving_square()
			m2 = time()
			solver.init_help_function(solve_segm = 'grad_desc')
			res_2 = solver.halving_square()
			m3 = time()
			results.append((eps, res_1[1], res_2[1], m2 - m1, m3 - m2))
			
		list_gss = [i[3] for i in results]
		list_grad = [i[4] for i in results]
		print(eps)
		print('GSS\n', 'Mean time = ', np.mean(list_gss))
		print('Failed tests =', len([i for i in results if i[1] < 0]))
		print('Gradient Descent\n', 'Mean time = ', np.mean(list_grad))
		print('Failed tests =', len([i for i in results if i[2] < 0]))
if __name__ == "__main__":
	eps = [0.1**(i) for i in range(3)]
	test_research_solve(eps)
