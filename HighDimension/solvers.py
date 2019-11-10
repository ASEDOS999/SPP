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

	def solver(self, ind, new_Q, method = 'HS', indexes = None, x_new = None):
		start_point = self.f.get_start_point(ind, new_Q)
		solver = self.methods[method]
		L_x,L = self.f.L[ind]
		if indexes is None:
			indexes = []
		indexes.append(new_Q[ind])
		def x_new_(x, Q = new_Q):
			if len(x) == len(Q):
				return x
			list_ = []
			k = 0
			for ind, i in enumerate(Q):
				if type(Q[ind]) == list:
					list_.append(x[k])
					k+=1
				else:
					list_.append(Q[ind])
			return np.array(list_)
		x_new = x_new_
		indexes = [i for i,_ in enumerate(new_Q) if type(_) != list]
		proj = lambda x: np.clip(x, *np.array(Q_).T)
		der = lambda x: self.f.get_grad(x_new(x), only_ind =  [ind])[0]
		grad = lambda x: self.f.get_grad(x_new(x), without = indexes)
		mu = self.f.mu[ind]
		cond = lambda x, size: self.CurGrad(x, der, size, L_x)
		kwargs = {'grad':grad,
				'start_point':start_point,
				'L':L,
				'mu':mu,
				'cond' : cond,
				'proj' : proj,
				'Q' : new_Q,
				'indexes' : indexes,
				'x_new' : x_new
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
		try:
			indexes = kwargs['indexes']
		except:
			indexes = None
		try:
			x_new = kwargs['x_new']
		except:
			x_new = None
		if not indexes is None and len(indexes)== len(self.Q)-1: # NEED MODIFICATION
			for i in Q:
				if type(i) == list:
					Q = i
					break
			a, b = Q[0],Q[1]
			size = (b-a)
			x = np.array([(b+a)/2])
			der = kwargs['grad']
			while not cond(x,size):
				if der(x) > 0:
					a, b = a,x[0]
				else:
					a,b = x[0], b
				size = (b-a)
				x = (b+a)/2
				x = np.array([x])
			return x
		n = 0
		while n==0 or (n < N and not cond(x,size)):
			n += 1
			ind = 0
			while ind < len(Q):
				if type(Q[ind]) == list:
					new_Q = Q.copy()
					new_Q[ind] = sum(Q[ind]) / 2
					g = self.solver(ind, new_Q, indexes = indexes, x_new = x_new)
					if g > 0:
						new_Q[ind] = [Q[ind][0], new_Q[ind]]
					else:
						new_Q[ind] = [new_Q[ind], Q[ind][1]]
					Q = new_Q
				ind += 1
			Q_ = np.array(Q)
			x = list()
			size = 0
			for i in Q:
				if type(i) == list:
					x.append(sum(i)/2)
					size += (i[1]-i[0])**2
				else:
					x.append(i)
			x = np.array(x)
			size = np.sqrt(size)
		return x


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
