import time

class TimeCount():
	def __init__(self):
		self.slot = {}
		self.count = {}
		self.clock = time.time()

	def start(self, key):
		self.slot[key] = time.time()
		if key not in self.count:
			self.count[key] = .0

	def check(self, key):
		if key in self.count:
			self.count[key] += time.time() - self.slot[key]

	def show(self):
		print("TIME SPEND:")
		for key, value in self.count.items():
			print('counter %s: time %f' % (str(key), value))

	def show_per_sec(self):
		a = time.time()
		if a - self.clock > 10.:
			self.show()
			self.clock = a