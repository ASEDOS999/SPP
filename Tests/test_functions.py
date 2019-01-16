# -*- coding: utf-8 -*-

from math import pi, cos
import numpy as np

class sinuses():
    def __init__(self, a, b):
        self.parameters = a
        self.coef = b
        self.min = self.calculate_function(b[0]/2, b[1]/2)
        self.min_x = b[0]/2
        self.min_y = b[1]/2
        s = a[0][1] + a[0][0]
        for i in range(1, len(a)):
            s += (i+1) * (a[i][0] + a[i][1]) * a[i][2]**i
        self.L = s
    def calculate_function(self, x, y):
        a = self.parameters
        b = self.coef
        z = -a[0][0] * np.sin(x * pi / b[0]) - a[0][1] * np.sin(y * pi / b[1])
        for i in range(1, len(a)):
            z +=  (-a[i][0] * np.sin(x * pi / b[0]) - a[i][1] * np.sin(y * pi / b[1]) + a[i][0] + a[i][1] + a[i][2])**(i+1)
        return z
    
    def der_x(self, x, y):
        a = self.parameters
        b = self.coef
        der = -a[0][0]
        for i in range(1, len(a)):
            der += (-(i+1) * a[i][0] * (-a[i][0] * np.sin(x * pi / b[0]) -
            a[i][1] * np.sin(y * pi / b[1]) + a[i][0] + a[i][1] + a[i][2])**i)
        return der * cos(x * pi / b[0]) * pi / b[0]
    
    def der_y(self, x, y):
        a = self.parameters
        b = self.coef
        der = -a[0][1]
        for i in range(1, len(a)):
            der += (-(i+1) * a[i][1] * (-a[i][0] * np.sin(x * pi / b[0]) -
            a[i][1] * np.sin(y * pi/b[1]) + a[i][0] + a[i][1] + a[i][2])**i)
        return der * cos(y * pi / b[1]) * pi / b[1]
    
    def gradient(self, x, y):
        return [self.der_x(x,y), self.der_y(x,y)]
    
    def get_est(self, x, num):
        return -1
    
#Quadratic function
class quadratic_function():
    def __init__(self, list_of_paremeters):
        self.A = list_of_paremeters
        
    def calculate_function(self, x, y):
        p = self.A
        f = p[0] * x**2 + 2 * p[1] * x * y + p[2] * y**2
        f += p[3] * x + p[4] * y + p[5]
        return f
    
    def der_x(self, x, y):
        p = self.A
        der = (p[3] + 2 * p[1] * y) + 2 * p[0] * x
        return der
    
    def der_y(self, x, y):
        p = self.A
        der = (p[4] + 2 * p[1] * x) + 2 * p[2] * y
        return der

    def gradient(self, x, y):
        return [self.der_x(x,y), self.der_y(x,y)]
    
    def ger_est(self, x, num):
        return 0