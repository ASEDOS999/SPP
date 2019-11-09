import numpy as np
class HalvingCube:
	def __init__(self, Q, f):
		self.Q = Q
		self.f = f
		self.methods = {
				'GD' : self.GD, 'HS': self.main
				}
	def GD(self, **kwargs):
		grad, x0, L, mu, cond, proj = (kwargs['grad'], kwargs['start_point'],
			kwargs['L'], kwargs['mu'], kwargs['cond'], kwargs['proj'])
		x, R = x0
		R0 = R
		R = R0
		k = 0
		M = L/mu
		alpha = 2/(L+mu)
		q = (M-1)/(M+1)
		while not (cond(x, R) or np.linalg.norm(grad(x))<=1e-15):
			x = (x - alpha * grad(x))
			R *= q
			k += 1
		return x
	
	def CurGrad(self, x, der, size, L):
		return abs(der(x)/L) >= size

	def solver(self, ind, new_Q, method = 'GD'):
		start_point = self.f.get_start_point(ind, new_Q)
		solver = self.methods[method]
		L_x,L = self.f.L[ind]
		x_new = lambda x: np.array([i for ind_, i in enumerate(x) if ind_ < ind] +
							  [new_Q[ind]] +
							  [i for ind_, i in enumerate(x) if ind_ >= ind])
		self.x_new = x_new
		Q_ = [i for ind_,i in enumerate(new_Q) if ind_!= ind]
		proj = lambda x: np.clip(x, *np.array(Q_).T)
		der = lambda x: self.f.get_grad(x_new(x))[ind]
		grad = lambda x: self.f.get_grad(x_new(x), without = ind)
		mu = self.f.mu[ind]
		cond = lambda x, size: self.CurGrad(x, der, size, L_x)
		kwargs = {'grad':grad,
				'start_point':start_point,
				'L':L,
				'mu':mu,
				'cond' : cond,
				'proj' : proj
			}
		x = solver(**kwargs)
		return der(x)

	def main(self, **kwargs):
		try:
			N = kwargs['N']
		except:
			N = np.inf
		try:
			Q = kwargs['Q']
		except:
			Q = self.Q
		try:
			cond = kwargs['cond']
		except:
			cond = lambda x, size: False
		n = 0
		while n==0 or (n < N and not cond(x,size)):
			n += 1
			ind = 0
			while ind < len(Q):
				new_Q = Q.copy()
				new_Q[ind] = sum(Q[ind]) / 2
				g = self.solver(ind, new_Q)
				if g > 0:
					new_Q[ind] = [Q[ind][0], new_Q[ind]]
				else:
					new_Q[ind] = [new_Q[ind], Q[ind][1]]
				Q = new_Q
				ind += 1
			Q_ = np.array(Q)
			x = np.mean(Q_, axis = 1)
			size = np.linalg.norm(Q_[:,1]-Q_[:,0])
		return np.array([sum(i)/2 for i in Q])


def gradient_descent(f, Q, N = 100):
	x0 = np.array([sum(i)/2 for i in Q])
	grad = lambda x: f.get_grad(x)
	L = f.L_full_grad
	x = x0
	n = 0
	while n < N:
		x = x - 0.5/L * grad(x)
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
