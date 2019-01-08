# -*- coding: utf-8 -*-

def grad_discent_segment(segm, deriv, alpha_0, delta):
    N = 1
    x, x_prev = segm[0], (segm[0] + segm[1]) / 2
    while abs(x - x_prev) > delta / 2:
        x = min(max(x_prev - alpha_0 / N * deriv(x_prev), segm[0]), segm[1])
        N += 1
    return x

def method(f, eps, Q):
    N = 1
    while True:
        N += 1
        x_0 = grad_discent_segment([Q[0], Q[1]], 
                                 lambda x: f.der_x(x, (Q[2] + Q[3]) / 2), 
                                 (Q[0] + Q[1]) / 4, 
                                 f.min_der_y((Q[2] + Q[3]) / 2) / f.max_der_xy((Q[2] + Q[3]) / 2, 1)))
        der = f.der_y(x_0, (Q[2] + Q[3]) / 2)
        if der == 0 or abs(f.calculate_function(x_0, (Q[2] + Q[3]) / 2) - f.min) <  eps:
            return ((x_0, (Q[2] + Q[3]) / 2), N)
        if der > 0:
            Q[2], Q[3] = Q[2],  (Q[2] + Q[3]) / 2
        else:
            Q[3], Q[2] = Q[3],  (Q[2] + Q[3]) / 2
        
        y_0 = grad_discent_segment([Q[2], Q[3]], 
                                 lambda y: f.der_y((Q[0] + Q[1]) / 2, y), 
                                 (Q[2] + Q[3]) / 4, 
                                 f.min_der_x((Q[0] + Q[1]) / 2) / f.max_der_xy((Q[0] + Q[1]) / 2, 0))))
        der = f.der_x((Q[0] + Q[1]) / 2, y_0)
        if der == 0 or abs(f.calculate_function((Q[0] + Q[1]) / 2, y_0) - f.min) <  eps:
            return (((Q[0] + Q[1]) / 2, y_0), N)
        if der > 0:
            Q[0], Q[1] = Q[0],  (Q[0] + Q[1]) / 2
        else:
            Q[1], Q[0] = Q[1],  (Q[0] + Q[1]) / 2