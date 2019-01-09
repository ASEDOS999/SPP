# -*- coding: utf-8 -*-

def grad_discent_segment(segm, deriv, alpha_0, delta, sol):
    if delta == 0:
        delta = 0.1**2
    N = 1
    x = (segm[0] + segm[1]) / 2

    while abs(x - sol) > delta and N < 1000:
        x = min(max(x - alpha_0 * deriv(x), segm[0]), segm[1])
        N += 1
    return (x, int(N) // 1000)

def method(f, eps, Q):
    N = 0
    flag = 0
    
    if abs(f.calculate_function((Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2) - f.min) <  eps:
        return (((Q[0] + Q[1]) / 2, (Q[2] + Q[3]) / 2), N, flag)
    while True:
        r = grad_discent_segment([Q[0], Q[1]], 
                                 lambda x: f.der_x(x, (Q[2] + Q[3]) / 2), 
                                 (Q[0] + Q[1]) / 4, 
                                 f.min_der_y((Q[2] + Q[3]) / 2) / f.max_der_xy((Q[2] + Q[3]) / 2, 1),
                                 f.min_x)
        x_0 = r[0]
        flag += r[1]
        der = f.der_y(x_0, (Q[2] + Q[3]) / 2)
        if der == 0 or abs(f.calculate_function(x_0, (Q[2] + Q[3]) / 2) - f.min) <  eps:
            return ((x_0, (Q[2] + Q[3]) / 2), N, flag)
        if der > 0:
            Q[2], Q[3] = Q[2],  (Q[2] + Q[3]) / 2
        else:
            Q[3], Q[2] = Q[3],  (Q[2] + Q[3]) / 2
        
        r = grad_discent_segment([Q[2], Q[3]], 
                                 lambda y: f.der_y((Q[0] + Q[1]) / 2, y), 
                                 (Q[2] + Q[3]) / 4, 
                                 f.min_der_x((Q[0] + Q[1]) / 2) / f.max_der_xy((Q[0] + Q[1]) / 2, 0),
                                 f.min_y)
        y_0 = r[0]
        flag += r[1]
        der = f.der_x((Q[0] + Q[1]) / 2, y_0)
        if der == 0 or abs(f.calculate_function((Q[0] + Q[1]) / 2, y_0) - f.min) <  eps:
            return (((Q[0] + Q[1]) / 2, y_0), N, flag)
        if der > 0:
            Q[0], Q[1] = Q[0],  (Q[0] + Q[1]) / 2
        else:
            Q[1], Q[0] = Q[1],  (Q[0] + Q[1]) / 2
        N += 1
        if N > 1000:
            return ((x_0, y_0), -1, flag)
