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
		name = ['Little Big', 'Constant', 'Stop_ineq']
		print('eps = ', "{:.1e}".format(eps))
		for j in range(3):
			list = [i[2 + j * 2] for i in results if i[1 + j * 2] >= 0]
			print('Mean time (%s) = %.2fms'%(name[j], 1000 * np.mean(list)))
			list = [i[2 + j * 2] for i in results if i[1 + j * 2] <  0]
			q = len(list)
			if q > 0:
				print("%s: Number of failed tests equals %d"%(name[j], q))
		full_results = full_results + results
		results = []

if __name__ == "__main__":
	eps = [0.1**(i) for i in range(7)]
	get_tests_estimates(eps)
