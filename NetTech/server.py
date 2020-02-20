import socket
import pickle

import sys
sys.path.append("../Tests")
sys.path.append("../Tests/Tests_functions")
sys.path.append("../Tests/Experiments")
import comparison, estimates

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = 'localhost'
port = 8007
s.bind((host, port))
s.listen(1)
conn, addr = s.accept()
print('client is at', addr)

while True:
	data = conn.recv(1000000)
	request = data.decode("utf8").split()
	if request[0] == "break":
		break
	elif request[0] == "LogSumExp":
		N, time_max, eps, C = None, None, None, None
		for i in request[1:]:
			args = i.split("=")
			if args[0] == "N":
				N = int(args[1])
			if args[0] == "time_max":
				time_max = float(args[1])
			if args[0] == "eps":
				eps = float(args[1])
			if args[0] == "C":
				C = float(args[1])
		if N is None:
			N = 10
		if time_max is None:
			time_max = 0.2
		if eps is None:
			eps = 1e-3
		if C is None:
			C = 1
		results, f = comparison.NEWcomparison_LogSumExp(N, time_max = time_max, eps = eps, C=C)
		keys = list(results.keys())
		ng = lambda x,y: f.calculate_function(x,y)
		new_dict = dict()
		for key in keys:
			times = [i - results[key][3][0] for i in results[key][3][:40]]
			f_value = [ng(i[0], i[1]) for i in results[key][2][:40]]
			new_dict[key] = (times, f_value)
		data = b""
		with open("1.pickle", "wb") as f:
			pickle.dump(new_dict, f)
			f.close()
		with open("1.pickle", "rb") as f:
			data = f.read()
			print(len(data))
			f.close()
	conn.send(data)
conn.close()
