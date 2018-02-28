import numpy as np
import math
from TestFPS import TestFPS
from VolleyBall import Volleyball, GameState, M, perspectiveTransform, mario_h, CENTER
from ObjectsTracker import objTracker
import cv2

from enum import Enum

PlayerState = Enum("PlayerState", "ADJUST POSITION DIRECT THROW WATCH")

class Rulebase():
	def __init__(self):
		self.game = Volleyball(serial=True)
		self.state = PlayerState.ADJUST
		self.action = "STAY"
		self.countFPS = TestFPS()

	def writeMove(self, direction, HEADER=""):
		x = int(math.cos(direction) * 128) + 128
		y = int(math.sin(direction) * 128) + 128
		if x == 256: x -= 1
		if y == 256: y -= 1
		return (HEADER + " " if HEADER else "") + ("%d %d" % (x, y))

	def play(self):
		try:
			while True:
				self.state = PlayerState.ADJUST
				if not self.game.wait_for_start():
					raise Exception('stop')
				print("start your performance")
				self.state = PlayerState.POSITION

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
					normal = False
					if p_player is not None and p_shadow is not None and p_ball is not None:
						normal = True
						p_player += np.array((0, mario_h))
						p_new_player = perspectiveTransform(p_player, M)
						p_new_shadow = perspectiveTransform(p_shadow, M)
						cv2.circle(frame, (p_player[0], p_player[1]), 2, (255,0,0), 2)
						cv2.circle(frame, (p_shadow[0], p_shadow[1]), 2, (0,255,0), 2)
						cv2.circle(frame, (p_ball[0], p_ball[1]), 2, (0,0,255), 2)
					else:
						p_player = p_shadow = p_ball = p_new_player = p_new_shadow = np.array((0,0))

					# cv2.imshow('nxt', frame)
					# if (cv2.waitKey(1) == ord('q')):
					# 	cv2.destroyAllWindows()
					# 	raise Exception("key interrupt")

					self.action = "STAY"
					if self.state == PlayerState.POSITION:
						if np.linalg.norm(p_shadow - p_ball) < 50.:
							self.state = PlayerState.DIRECT
							if np.linalg.norm(CENTER - p_new_player) < 5.:
								direction = math.atan2(*(CENTER - p_new_player))
								self.action = self.writeMove(direction, "MOVE")
							self.countFPS.save()
					elif self.state == PlayerState.DIRECT:
						if self.countFPS.frames() < 10:
							direction = math.atan2(*(p_new_shadow - p_new_player))
							self.action = self.writeMove(direction, "MOVE")
						else:
							self.state = PlayerState.THROW
							self.countFPS.save()
					elif self.state == PlayerState.THROW:
						if self.countFPS.frames() < 2:
							self.action = "THROW"
						else:
							self.state = PlayerState.WATCH
					elif self.state == PlayerState.WATCH:
						if self.score < score:
							self.state = PlayerState.WATCH
							self.countFPS.save()
					self.score = score
		except:
			import sys, os
			self.game.close()
			exc_type, exc_obj, exc_tb = sys.exc_info()
			fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
			print(exc_type, fname, exc_tb.tb_lineno)	

if __name__ == '__main__':
	player = Rulebase()
	player.play()