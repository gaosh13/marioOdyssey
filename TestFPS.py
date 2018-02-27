import time

class TestFPS:
	def __init__(self):
		self.stime = .0
		self.scount = 0
		self.count = 0

	def save(self):
		self.stime = time.time()
		self.scount = self.count

	def plus(self):
		self.count += 1

	def fps(self):
		a = time.time()
		return (self.count - self.scount) / (a - self.stime)

	def frames(self):
		return self.count - self.scount

	def print_per_sec(self):
		a = time.time()
		if a - self.stime > 1.:
			print("FPS:", self.fps())
			self.save()