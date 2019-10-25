import numpy as np
class HalvingCube:
	def __init__(self, Q, f):
		self.Q = Q
		self.f = f
		self.methods = {
				'GD' : self.GD
				}
	def GD(grad, x0, L, mu, cond):
		x, R = x0
		R0 = R
		k = 0
		if mu > 0:
			M = L/mu
			q = (M-1)/(M+1)
		else:
			q = np.inf
		while not cond(x, R):
			x = x - 1/L * grad(x)
			R = R0/(k+4) if R0/(k+4) < R * q else R * q
			k += 1
		return x
	
	def CurGrad(self, x, der, size, L):
		return der(x)/L > size

	def solver(self, ind, new_Q, method = 'GD'):
		start_point = self.f.get_start_point(ind, new_Q)
		solver = self.methods[method]
		L_x,L = self.f.L[ind]
		x_new = lambda x: np.array([i for ind_, i in enumerate(x) if ind_ < ind] +
							  new_Q[ind] +
							  [i for ind_, i in enumerate(x) if ind_ >= ind])
		der = lambda x: self.f.grad[ind](x_new(x))
		grad = lambda x: self.f.get_grad(x, without = ind)
		mu = self.f.mu[ind]
		cond = lambda x, size: self.CurGrad(x, der, size, L_x)
		return der(solver(grad, start_point, L, mu, cond))

	def main(self, Q = self.Q, d = len(self.Q), N = 100):
		n = 0
		while n < N:
			for ind,i in enumerate(Q):
				new_Q = Q.copy()
				new_Q[ind] = sum(Q[ind]) / 2
				g = self.solver()
				if g > 0:
					Q = [Q[ind][0], new_Q[ind]]
				else:
					Q = [new_Q[ind], Q[ind][1]]
		return np.array([sum(i)/2 for i in Q])


def gradient_descent(f, Q, grad, L, eps, minimum, N = 100):
	N = 0
	x0 = np.array([sum(i) for i in Q])
	grad = lambda x: f.get_grad(x)
	L = sum(f.L_list)
	x = x0
	n = 0
	while n < N:
		x = x - 1/L * grad(x)
		n += 1
	return x

def ellipsoid(f, Q, x_0=None, eps=None, N = 100):
	n = len(Q)
	x0 = np.array([sum(i) for i in Q])
	x = x0
	eps = 5e-3 if eps is None else eps
	rho = (Q[1] - Q[0]) * np.sqrt(n) / 2
	H = np.identity(n)
	domain = np.array([[Q[0], Q[1]], [Q[2], Q[3]]])
	k = 0
	results = [x]
	while k < N:
		gamma = (rho / (n+1)) * (n / np.sqrt(n ** 2 - 1)) ** k
		_df = f.get_grad(x)
		_df = _df / (np.sqrt(abs(_df@H@_df)))
		x = np.clip(x - gamma * H @ _df, *domain.T)
		H = H - (2 / (n + 1)) * (H @ np.outer(_df, _df) @ H)
		k += 1
		results.append(x)
	if k >= 100:
		k = -1
	return (x, k, results)