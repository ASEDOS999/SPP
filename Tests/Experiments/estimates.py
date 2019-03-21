import sys

sys.path.append('..')
sys.path.append('../Tests_functions')

from time import time
import matplotlib.pyplot as plt
import numpy as np
import random
from test_functions import quadratic_function as qf
from method_functions import main_solver
import math

def get_tests_estimates(epsilon):
	#Quadratic functions
	results = []
	num = 0
	N = 1000
	n = 0
	full_results = []
	for eps in epsilon:
		for j in range(N):
			param = np.random.uniform(-10, 10, 6)
			param[2] =  abs(param[2])
			f = qf(param)
			size_1, size_2 = random.uniform(0.5, 1), random.uniform(0.5, 1)
			x_1, y_1 = f.solution[0], f.solution[1]
			Q = [x_1 - (1-size_1), x_1 + (1+size_1), y_1 - (1-size_2), y_1 + (1+size_2)]
			R = 2
			L = f.lipschitz_function(Q)
			M = f.lipschitz_gradient(Q)
			solver = main_solver(f, Q, eps)
			m1 = time()
			solver.init_help_function(stop_func = 'big_grad')
			res_1 = solver.halving_square()
			m2 = time()
			solver.init_help_function(stop_func = 'const_est')
			res_2 = solver.halving_square()
			m3 = time()
			solver.init_help_function(stop_func = 'stop_ineq')
			res_3 = solver.halving_square()
			m4 = time()
			results.append((eps, res_1[1], m2-m1, res_2[1], m3-m2, res_3[1], m4 - m3))
		list_gss = [i[2] for i in results]
		print('eps = ', eps)
		print('Mean time (Little Big) = %.2fms'%(1000 * np.mean([i[2] for i in results])))
		print('Mean time (Constant) = %.2fms'%(1000 * np.mean([i[4] for i in results])))
		print('Mean time (Stop_ineq) = %.2fms'%(1000 * np.mean([i[6] for i in results])))
		full_results = full_results + results
		results = []

if __name__ == "__main__":
	eps = [0.1**(i) for i in range(9)]
	get_tests_estimates(eps)
