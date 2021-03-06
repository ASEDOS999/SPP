# -*- coding: utf-8 -*-

import sys
sys.path.append("/Tests_functions")

from time import time
import matplotlib.pyplot as plt
import numpy as np
import random
from test_functions import sinuses
from test_functions import LogSumExp
from test_functions import quadratic_function as qf
from method_functions import main_solver
from method_functions import halving_square as HS
from method_functions import gradient_descent
from method_functions import ellipsoid
from method_functions import FGM
import math

def qf_comparison(epsilon = 1e-6, out = True):
	param = np.random.uniform(-100, 100, 6)
	param[2] =  abs(param[2])
	f = qf(param)
	size_1, size_2 = random.uniform(0.5, 1), random.uniform(0.5, 1)
	x_1, y_1 = f.solution[0], f.solution[1]
	Q = [x_1 - (1-size_1), x_1 + (1+size_1), y_1 - (1-size_2), y_1 + (1+size_2)]
	results = comparison(f, Q, epsilon)
	if out:
		norm_gradient = lambda x: f.calculate_function(x[0], x[1]) - f.min
		plt.semilogy([i for i in range(len(results[0]))], [norm_gradient(i) for i in results[0]])
		plt.semilogy([i for i in range(len(results[2]))], [norm_gradient(i) for i in results[2]])
		plt.semilogy([i for i in range(len(results[4]))], [norm_gradient(i) for i in results[4]])
		plt.xlabel('Number of iterations')
		plt.ylabel(r'$f(x)-f^*$')
		plt.xticks([i for i in range(0, 101, 10)])
		plt.legend(['Gradiend Descent', 'Halving Square', 'Ellipsoid'])
		print('Gradiend Descent %.4f'%(results[1]))
		print('Halving Square %.4f'%(results[3]))
		print('Ellipsoid %.4f'%(results[5]))
	return results

def qf_test(eps):
	param = np.random.uniform(-100, 100, 6)
	param[2] =  abs(param[2])
	f = qf(param)
	a = 1
	size_1, size_2 = random.uniform(0.5 *a, a), random.uniform(0.5*a, a)
	x_1, y_1 = f.solution[0], f.solution[1]
	Q = [x_1 - (a-size_1), x_1 + (a+size_1), y_1 - (a-size_2), y_1 + (a+size_2)]
	f.lipschitz_function(Q), f.lipschitz_gradient(Q)
	solver = main_solver(f, Q, eps)
	solver.init_help_function()
	res = solver.halving_square()[2]
	norm_gradient = lambda x: f.calculate_function(x[0], x[1]) - f.min
	plt.semilogy([i for i in range(len(res))], [norm_gradient(i) for i in res])
	plt.semilogy([i for i in range(len(res))], [(i[0] -x_1)**2 + (i[1]-y_1)**2 for i in res])
	plt.grid()
	n = 13
	plt.legend([r"$f(x)-f^*$", r"$\|x-x^*\|$"], fontsize = n)
	q_2 = 1./2 * np.log(f.M * a**2/(4 * eps)) / np.log(2)
	q_1 = np.log(f.L * a/(np.sqrt(2) * eps)) / np.log(2)
	plt.xlabel("Iterations Number", fontsize = n)
	plt.ylabel("Value of Error", fontsize = n)
	plt.xticks(fontsize = n)
	plt.yticks(fontsize = n)
	print('Theoretical Iteration Number through function constant', np.ceil(q_1))
	print('Theoretical Iteration Number through gradient constant', np.ceil(q_2))
	return res

def qf_test_2(eps):
	param = [1, 1, 1, 0, 0, 0]
	f = qf(param)
	a = 1
	size_1, size_2 = random.uniform(0.5 *a, a), random.uniform(0.5*a, a)
	x_1, y_1 = 1, 1
	f.solution[0], f.solution[1] = x_1, y_1
	f.min = 5
	Q = [1, 2, 1, 2]
	f.lipschitz_function(Q), f.lipschitz_gradient(Q)
	solver = main_solver(f, Q, eps)
	solver.init_help_function()
	res = solver.halving_square()[2]
	norm_gradient = lambda x: f.calculate_function(x[0], x[1]) - f.min
	plt.semilogy([i for i in range(len(res))], [norm_gradient(i) for i in res])
	plt.semilogy([i for i in range(len(res))], [(i[0] -x_1)**2 + (i[1]-y_1)**2 for i in res])
	plt.grid()
	n = 13
	plt.xlabel("Iterations Number", fontsize = 13)
	plt.ylabel("Value of Error", fontsize = 13)
	plt.legend([r"$f(x)-f^*$", r"$\|x-x^*\|$"], fontsize = 13)
	plt.xticks(fontsize = 13)
	plt.yticks(fontsize = 13)
	q_2 = 1./2 * np.log(f.M * a**2/(4 * eps)) / np.log(2)
	q_1 = np.log(f.L * a/(np.sqrt(2) * eps)) / np.log(2)
	print('Theoretical Iteration Number through function constant', np.ceil(q_1))
	print('Theoretical Iteration Number through gradient constant', np.ceil(q_2))
	return res

def comparison(f, Q, eps):
	n = 0
	m1 = time()
	res_1 = gradient_descent(f.calculate_function, Q, f.gradient, f.lipschitz_gradient(Q) ,eps, f.min)
	m2 = time()
	solver = main_solver(f, Q, eps)
	solver.init_help_function()
	res_2 = solver.halving_square()
	m3 = time()
	res_3 = ellipsoid(f, Q, eps = eps)
	m4 = time()
	return res_1[2], m2-m1, res_2[2], m3-m2, res_3[2], m4-m3

import threading
def NEWcomparison_LogSumExp(N = 2, time_max = 100, a = None, eps = 0.001, C = 1):
	if a is None:
		k = 1000
		a = np.random.uniform(-k, k, size=(N,))
	f = LogSumExp(a, C = C)
	Q = f.get_square()
	L, M = f.lipschitz_function(Q), f.lipschitz_gradient(Q)
	res = dict()
	fdict = dict()
	
	
	print('Ellipsoids')
	t1 = threading.Thread(target = ellipsoid, 
						args = (f,Q.copy()),
						kwargs = {'cur_eps':eps, 'time_max':time_max, 'time': True, 'res':(res,'Ellipsoids')})
	
	solver = HS(f,Q,None, cur_eps = eps)
	print('CurGrad')
	t2 = threading.Thread(target = solver.halving_square, 
						kwargs = {'time_max':time_max, 'time': True, 'res':(res, 'HalvingSquare-CurGrad')})
	res['HalvingSquare-CurGrad']= None

	solver = HS(f,Q,None, cur_eps = eps)
	print('ConstEst')
	solver.stop = solver.ConstEst
	t3 = threading.Thread(target = solver.halving_square,
						kwargs = {'time_max':time_max, 'time': True, 'res':(res,'HalvingSquare-ConstEst')})
	
	print('PGM')
	t4 = threading.Thread(target = gradient_descent, 
						args = (f, Q, M), kwargs={'time':True, 'time_max':time_max, 'res':(res,'PGM'), 'cur_eps':eps})

	print('FGM')
	t5 = threading.Thread(target = FGM, 
						args = (f, Q, M), kwargs={'time':True, 'time_max':time_max, 'res':(res,'FGM'), 'cur_eps':eps})
	
	t1.start()
	t2.start()
	t3.start()
	t4.start()
	t5.start()
	t1.join()
	t2.join()
	t3.join()
	t4.join()
	t5.join()
	keys =res.keys()
	for key in keys:
		fdict = {**fdict, **res[key][-1]}
		res[key] = tuple([i for i in res[key][:-1]])
	f.values = fdict
	return res, f

def strategy_LogSumExp(N = 2, time_max = 100, eps = 1e-3, a = None, f = None):
	if f is None:
		if a is None:
			k = 1000
			a = np.random.uniform(-k, k, size=(N,))
		f = LogSumExp(a)
	Q = f.get_square()
	L, M = f.lipschitz_function(Q), f.lipschitz_gradient(Q)
	res = dict()
	fdict = dict()
	
	solver = HS(f,Q,None)
	solver.stop = solver.CurGrad
	N_max = 10
	print('CurGrad')
	t1 = threading.Thread(target = solver.halving_square, 
						kwargs = {'N_max':N_max, 'time': True, 'res':(res, 'HalvingSquare-CurGrad')})
	res['HalvingSquare-CurGrad']= None

	print('ConstEst')
	t2 = list()

	eps_list = [0.1**i for i in range(10)]
	for eps in eps_list:
		solver = HS(f,Q,None,cur_eps = eps)
		solver.stop = solver.ConstEst
		solver.eps = eps
		solver.est = None
		res[eps] = None
		t2.append(threading.Thread(target = solver.halving_square,
							kwargs = {'N_max':N_max, 'time': True, 'res':(res,eps)}))
	
	t1.start()
	for t in t2:
		t.start()
	t1.join()
	for t in t2:
		t.join()
	keys =res.keys()
	for key in keys:
		fdict = {**fdict, **res[key][-1]}
		res[key] = tuple([i for i in res[key][:-1]])
	f.values = fdict
	return res, f

def comparison_LogSumExp(N = 2, time_max = 100, a = None):
	if a is None:
		a = np.random.uniform(-100, 100, size=(N,))
	f = LogSumExp(a)
	Q = f.get_square()
	L, M = f.lipschitz_function(Q), f.lipschitz_gradient(Q)
	solver = main_solver(f, Q, eps = None)
	res = dict()
	fdict = dict()
	
	print('Ellipsoids')
	res['Ellipsoids'] = ellipsoid(f,Q, time_max = time_max, time = True, cur_eps = eps)
	fdict = {**fdict, **f.values}
	f.values = dict()
	
	print('CurGrad')
	solver.init_help_function()
	res['HalvingSquare-CurGrad']= solver.halving_square(time_max = time_max, time = True)
	fdict = {**fdict, **f.values}
	f.values = dict()
	
	print('Const_est')
	solver.init_help_function('const_est')
	res['HalvingSquare-Const']= solver.halving_square(eps = 1e-2)
	fdict = {**fdict, **f.values}
	f.values = dict()
	
	print('GD')
	res['GD'] = gradient_descent(f.calculate_function, Q, f.gradient, M, time= True, time_max = time_max)
	fdict = {**fdict, **f.values}
	f.values = fdict
	
	return res, f
if __name__ == "__main__":
	eps = [0.1**(i) for i in range(7)]
	num_iter_tests(eps)
	comparison_GD_HS_sinuses(eps)
	comparison_GD_HS_QFunc(eps)
