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

def get_tests_estimates(epsilon, mean_input = True, num_compare = False):
	N = 1000
	name = ['True gradient', 'Constant estimate', 'Current gradient']
	full_results = []
	for eps in epsilon:
		results = []
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
			solver.init_help_function(stop_func = 'true_grad')
			res_1 = solver.halving_square()
			m2 = time()
			solver.init_help_function(stop_func = 'const_est')
			res_2 = solver.halving_square()
			m3 = time()
			solver.init_help_function(stop_func = 'cur_grad')
			res_3 = solver.halving_square()
			m4 = time()
			results.append((eps, res_1[1], m2-m1, res_2[1], m3-m2, res_3[1], m4 - m3))
		if mean_input:
			print('eps = ', "{:.1e}".format(eps))
			for j in range(3):
				list = [i[2 + j * 2] for i in results if i[1 + j * 2] >= 0]
				print('Mean time (%s) = %.2fms'%(name[j], 1000 * np.mean(list)))
				list = [i[2 + j * 2] for i in results if i[1 + j * 2] <  0]
				q = len(list)
				if q > 0:
					print("%s: Number of failed tests equals %d"%(name[j], q))
		if num_compare:
			list_n = [(i, j, k) for i in range(3) for j in range(3) for k in range(3) if i != j and j != k and i != k]
			for n in list_n:
				ind = len([i for i in results if i[2 + n[0] * 2] <= i[2 + n[1] * 2] <= i[2 + n[2] * 2]])
				print("'%s' <= '%s' <= '%s': %d"%(name[n[0]], name[n[1]], name[n[2]], ind))
		full_results += results
	return full_results

if __name__ == "__main__":
	eps = [0.1**(i) for i in range(7)]
	get_tests_estimates(eps, num_compare = True)
