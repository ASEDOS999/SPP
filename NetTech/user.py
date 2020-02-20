import socket
import pickle
import matplotlib.pylab as plt

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '10.55.169.240'
port = 55088
#host = ''
#port = 8007
s.connect((host, port))

while True:
	i = input()
	if i == "break":
		s.send(bytes(i.encode("utf8")))
		break
	name_file = i.split()[-1]
	s.send(bytes(i.encode("utf8")))
	data = s.recv(1000000)
	if i.split()[0] == "LogSumExp":
		with open("res.pickle", "wb") as f:
			f.write(data)
			f.close()
		with open("res.pickle", "rb") as f:
			new_dict = pickle.load(f)
			f.close()
		keys = new_dict.keys()
		for key in keys:
			plt.plot(new_dict[key][0], new_dict[key][1])
			plt.xlabel("Time")
			plt.ylabel("Dual Problem")
			plt.grid()
		plt.legend(keys)
		plt.show()
	print('received', len(data), 'bytes')
s.close()
