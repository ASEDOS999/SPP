# -*- coding: utf-8 -*-

import math

def grad_descent_segment(segm, deriv, delta):
    N = 0
    x, x_prev, alpha_0 = (segm[0] + segm[1]) / 2, (segm[0] + segm[1]) / 2, (segm[0] + segm[1]) / 4
    if delta == 0:
        delta = 0.01 * alpha_0
        #delta = -1
    if delta < 0:
        return (x, N)
    while (abs(x - x_prev) > delta and N < 1000) or (N == 0):
        x, x_prev = min(max(x - alpha_0 / math.sqrt(N + 1) * deriv(x), 
                            segm[0]), 
                        segm[1]), x
        N += 1
    if N >= 1000:
        N = -1
    return (x, N)

def halving_square(f, eps, square):
    Q = square.copy()
    N = 0
    minimum = f.min
    x_0, y_0 = (Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2
    if f.der_x(x_0, y_0) == 0 and f.der_y(x_0, y_0) == 0:
        return ((x_0, y_0), N)
    f_opt = f.calculate_function(x_0, y_0)
    while True:
        x_0 = grad_descent_segment([Q[0], Q[1]], 
                                 lambda x: f.der_x(x, (Q[2] + Q[3]) / 2),
                                 f.get_est((Q[2] + Q[3]) / 2, 1))[0]
        der = f.der_y(x_0, (Q[2] + Q[3]) / 2)
        if der == 0 and f.der_x(x_0, (Q[2] + Q[3]) / 2) == 0:
            return ((x_0, (Q[2] + Q[3]) / 2), N)
        if der > 0:
            Q[2], Q[3] = Q[2],  (Q[2] + Q[3]) / 2
        else:
            Q[3], Q[2] = Q[3],  (Q[2] + Q[3]) / 2
        
        y_0 = grad_descent_segment([Q[2], Q[3]], 
                                 lambda y: f.der_y((Q[0] + Q[1]) / 2, y),
                                 f.get_est((Q[0] + Q[1]) / 2, 0))[0]
        der = f.der_x((Q[0] + Q[1]) / 2, y_0)
        if der == 0 and f.der_y((Q[0] + Q[1]) / 2, y_0) == 0:
            return (((Q[0] + Q[1]) / 2, y_0), N)
        if der > 0:
            Q[0], Q[1] = Q[0],  (Q[0] + Q[1]) / 2
        else:
            Q[1], Q[0] = Q[1],  (Q[0] + Q[1]) / 2
        N += 1
        
        x_0, y_0 = (Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2
        f_opt = f.calculate_function(x_0, y_0) 
        if N >= 1000 or abs(f_opt - minimum) < eps:
            if N >= 1000:
                N = -1
            return ((x_0, y_0), N)
        
def gradient_descent(f, Q, grad, eps, step, minimum):
    N = 0
    x = [(Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2]
    x_prev = [(Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2]
    while (abs(f(x[0], x[1]) - f(x_prev[0], x_prev[1])) > eps and N < 1000) or (N == 0):
        der = grad(x[0], x[1])
        x[0], x_prev[0] = min(max(x[0] - step / math.sqrt(N+1) * der[0], Q[0]), Q[1]), x[0]
        x[1], x_prev[1] = min(max(x[1] - step / math.sqrt(N+1) * der[1], Q[2]), Q[3]), x[1]
        N += 1
    if N >= 1000:
        N = -1
    return (x, N, abs(f(x[0], x[1]) - f(x_prev[0], x_prev[1])))