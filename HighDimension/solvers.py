import numpy as np
class HalvingCube:
	def __init__(self, Q, f):
		self.Q = Q
		self.f = f
		self.methods = {
				'HS': self.main
				}
	
	def condition_for_condition(self):
		#...
	def condition_for_step(self):
		#...
	
	def CurGrad(self, lambda_, der, size, L):
		return abs(der(lambda_)/L) >= size

	def solver(self, ind, new_Q, method = 'HS', indexes = None, lambda_new = None):
		start_point = None
		solver = self.methods[method]
		L_lambda,L = self.f.L[ind]
		if indexes is None:
			indexes = []
		else:
			indexes = indexes.copy()
		indexes.append(new_Q[ind])
		def lambda_new_(x, Q = new_Q):
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
		lambda_new = lambda_new_
		indexes = [i for i,_ in enumerate(new_Q) if type(_) != list]
		proj = lambda lambda_: np.clip(lambda_, *np.array(Q_).T)
		der = lambda lambda_: self.f.get_grad(lambda_new(lambda_), only_ind =  [ind])[0]
		grad = lambda lambda_: self.f.get_grad(lambda_new(lambda_), without = indexes)
		mu = self.f.mu[ind]
		cond = lambda lambda_, size: self.CurGrad(lambda_, der, size, L_lambda)
		kwargs = {'grad':grad,
				'start_point':start_point,
				'L':L,
				'mu':mu,
				'cond' : cond,
				'proj' : proj,
				'Q' : new_Q,
				'indexes' : indexes,
				'lambda_new' : lambda_new
			}
		lambda_ = solver(**kwargs)
		return der(lambda_)

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
			if 'eps' in kwargs:
				eps = kwargs['eps']
				cond = lambda lambda_, size: np.linalg.norm(self.f.get_grad(lambda_))<=eps
			else:
				cond = lambda lambda_, size: False
		try:
			indexes = kwargs['indexes']
		except:
			indexes = None
		if not indexes is None and len(indexes)== len(self.Q)-1: # NEED MODIFICATION
			for i in Q:
				if type(i) == list:
					Q = i
					break
			a, b = Q[0],Q[1]
			size = (b-a)
			lambda_ = np.array([(b+a)/2])
			der = kwargs['grad']
			while not cond(lambda_,size):
				if der(lambda_) > 0:
					a, b = a,lambda_[0]
				else:
					a, b = lambda_[0], b
				size = (b-a)
				lambda_ = (b+a)/2
				lambda_ = np.array([lambda_])
			return lambda_
		n = 0
		while n==0 or (n < N and not cond(lambda_,size)):
			n += 1
			ind = 0
			while ind < len(Q):
				if type(Q[ind]) == list:
					new_Q = Q.copy()
					new_Q[ind] = sum(Q[ind]) / 2
					g = self.solver(ind, new_Q, indexes = indexes)
					if g > 0:
						new_Q[ind] = [Q[ind][0], new_Q[ind]]
					else:
						new_Q[ind] = [new_Q[ind], Q[ind][1]]
					Q = new_Q
				ind += 1
			Q_ = np.array(Q)
			lambda_ = list()
			size = 0
			for i in Q:
				if type(i) == list:
					lambda_.append(sum(i)/2)
					size += (i[1]-i[0])**2
				else:
					lambda_.append(i)
			lambda_ = np.array(lambda_)
			size = np.sqrt(size)
		return lambda_


def gradient_descent(f, Q, N = 100, eps = None):
	x0 = np.array([sum(i)/2 for i in Q])
	grad = lambda x: f.get_grad(x)
	L = f.L_full_grad
	x = x0
	n = 0
	while n < N and (eps is None or np.linalg.norm(f.get_grad(x))>=eps):
		x = x - 0.5/L * grad(x)
		n += 1
	return x

def ellipsoid(f, Q, x_0=None, eps=None, N = 100):
	n = len(Q)
	x0 = np.array([sum(i)/2 for i in Q])
	x = x0
	eps = 5e-3 if eps is None else eps
	rho = (Q[0][1] - Q[0][0]) * np.sqrt(n) / 2
	H = np.identity(n)
	domain = np.array(Q)
	k = 0
	results = [x]
	while k < N and (eps is None or np.linalg.norm(f.get_grad(x))>=eps):
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
