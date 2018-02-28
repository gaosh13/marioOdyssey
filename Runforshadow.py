import numpy as np
import math
from TestFPS import TestFPS
from VolleyBall import Volleyball, GameState, M, perspectiveTransform, mario_h, CENTER
from ObjectsTracker import objTracker
import cv2

from enum import Enum

PlayerState = Enum("PlayerState", "ADJUST POSITION DIRECT THROW WATCH")

class Runforshadow():
	def __init__(self):
		self.game = Volleyball(serial=True)
		self.countFPS = TestFPS()
		self.pList = []
		self.sList = []
		self.bList = []
		self.action = 0
		self.md = 0
		
	def game_init(self):
		self.pList = [np.array((0,0)) for i in range(20)]
		self.sList = [np.array((0,0)) for i in range(20)]
		self.bList = [np.array((0,0)) for i in range(20)]
		self.his = [0 for i in range(20)]
		self.cms = [0, 0, 0]
		self.action = 0
		self.md = 0

	def calc(self, lst):
		for i in range(-5, 0):
			if lst[i][0] != 0:
				return np.int16(((-1-i)*lst[-1] + lst[i]) / (-i))
		return lst[-1]

	def disc(self, dx, dy):
		if max(abs(dx), abs(dy)) < 20:
			self.md = 0
		else:
			self.md = 0

	def cpxy(self, x, y):
		if self.his.count(2) + self.his.count(4) > 15:
			return False
		if self.his.count(1) + self.his.count(3) > 12:
			return True
		if x < 0:
			x = x * -0.3
		else:
			x = x * 15
		if y < 0:
			y = y * -1
		return x > y

	def play(self):
		try:
			while True:
				if not self.game.wait_for_start():
					raise Exception('stop')
				print("start your performance")
				self.game_init()

				while True:
					alive, state, score, frame = self.game.step(self.action)
					self.countFPS.plus()
					if not alive:
						raise Exception('stop')
					if state in (GameState.OVER, GameState.HANG):
						break

					centers, frame = objTracker.update(frame)
					p_player = centers[0]
					p_shadow, p_ball = centers[1], centers[2]

					if p_player is None:
						p_player = np.array((0, 0))
					if p_shadow is None:
						p_shadow = np.array((0, 0))
					if p_ball is None:
						p_ball = np.array((0, 0))
					
					if self.cms[0] < 10 and self.pList[-1][0] != 0 and np.linalg.norm(self.pList[-1] - p_player) > 30.:
						p_player = np.array(self.pList[-1])
						# print(self.pList)
						self.cms[0] += 1
					else:
						self.cms[0] = 0
					if self.cms[1] < 10 and self.bList[-1][0] != 0 and np.linalg.norm(self.bList[-1] - p_ball) > 30.:
						p_ball = self.calc(self.bList)
						self.cms[1] += 1
					else:
						self.cms[1] = 0
					if self.cms[2] < 10 and self.sList[-1][0] != 0 and np.linalg.norm(self.sList[-1] - p_shadow) > 30.:
						p_shadow = self.calc(self.sList)
						self.cms[2] += 1
					else:
						self.cms[2] = 0
					
					self.pList.append(np.array(p_player))
					self.pList.pop(0)
					self.sList.append(p_shadow)
					self.sList.pop(0)
					self.bList.append(p_ball)
					self.bList.pop(0)
					normal = False
					# print(centers, p_player, p_shadow, p_ball)
					if p_player[0] != 0 and p_shadow[0] != 0 and p_ball[0] != 0:
						normal = True
						p_player += np.array((0, mario_h))
						p_new_player = perspectiveTransform(p_player, M)
						p_new_shadow = perspectiveTransform(p_shadow, M)
						cv2.circle(frame, (p_player[0], p_player[1]), 2, (255,0,0), 4)
						cv2.circle(frame, (p_shadow[0], p_shadow[1]), 2, (0,255,0), 4)
						cv2.circle(frame, (p_ball[0], p_ball[1]), 2, (0,0,255), 4)
					else:
						p_player = p_shadow = p_ball = p_new_player = p_new_shadow = np.array((0,0))

					cv2.imshow('nxt', frame)
					if (cv2.waitKey(1) == ord('q')):
						cv2.destroyAllWindows()
						raise Exception("key interrupt")
					self.action = 0
					if normal:
						dx = p_new_shadow[0] - p_new_player[0]
						dy = p_new_shadow[1] - p_new_player[1]
						if dx * dx + dy * dy < 50:
							self.action = 0
						elif self.md != 0:
							self.md -= 1
						else:
							self.disc(dx, dy)
							if self.cpxy(dx, dy):
								if dx > 0:
									self.action = 4
								else:
									self.action = 2
							else:
								if dy > 0:
									self.action = 3
								else:
									self.action = 1
					self.his.append(self.action)
					self.his.pop(0)
		except:
			import sys, os
			self.game.close()
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)

if __name__ == '__main__':
	player = Runforshadow()
	player.play()