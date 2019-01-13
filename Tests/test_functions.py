# -*- coding: utf-8 -*-

from math import pi, sqrt, cos
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
    
    def get_est(self, x, num):
        return -1