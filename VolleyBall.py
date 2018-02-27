import numpy as np
import cv2
import time
import json
from TestFPS import TestFPS
from TimeCount import TimeCount
from Arduino_serial import Serial, Serial_null
from enum import Enum

Width = 640
Height = 360

# SCORE = {
#     'l2': 530, 'r2': 590, 't2': 240, 'b2': 290, 'line': 10,
#     'l1': 520, 'r1': 600, 't1': 230, 'b1': 300,
# }
# MISS = {
#     'l2': 290, 'r2': 370, 't2': 140, 'b2': 160, 'line': 10,
#     'l1': 280, 'r1': 380, 't1': 130, 'b1': 170,
# }

SCORE = {'l2': 530, 'r2': 590, 't2': 240, 'b2': 290}
MISS = {'l2': 290, 'r2': 370, 't2': 140, 'b2': 160}
ALL = {'l2': 0, 'r2': 640, 't2': 0, 'b2': 360}

SCORE_ZERO_FILE = "score_pixel"
MISS_FILE = "miss"

unpack = lambda dict, *args: (dict[arg] for arg in args)
toclass = lambda cls, lcs, *args: [setattr(cls, arg, lcs[arg]) for arg in args]

lower_green = np.array([45,150,40])
upper_green = np.array([57,210,220])
field = np.array([[240, 145], [190, 290], [435, 290], [390, 145]])
sqrField = np.array([[0, 0], [0, 100], [77, 100], [77, 0]])
CENTER = np.array([39, 50])
MLEN = np.linalg.norm(CENTER)
M = cv2.getPerspectiveTransform(np.float32(field),np.float32(sqrField))

w_ex = 50
w_bl = 500
w_in = 20
fieldA = np.array(((240-w_ex,145-w_bl), (390+w_ex,145-w_bl), (390+w_ex, 145+w_in), (240-w_ex, 145+w_in))) # +20
fieldB = np.array(((190-w_ex,290-w_in), (435+w_ex,290-w_in), (435+w_ex, 290+w_bl), (190-w_ex, 290+w_bl))) # -10
fieldC = np.array(((372-w_in, 87), (372+w_bl, 87), (453+w_bl, 348), (453-w_in, 348))) # -10
b_in = 10
boundField = np.array([0+b_in, 77-b_in, 0+b_in, 100-b_in])

mario_h = 25

# testFPS
countFPS = None
arduino_serial = None

GameState = Enum("GameState", "LEAVE ENTER WAIT LIVE OVER HANG")

# def draw_frame(frame, bound, color=np.array([255, 0 , 0])):
#     l1, r1, t1, b1, l2, r2, t2, b2 = unpack(bound, 'l1', 'r1', 't1', 'b1', 'l2', 'r2', 't2', 'b2')
#     for i in range(Height):
#         if not bound['t1'] <= i < bound['b1']:
#             continue
#         for j in range(Width):
#             if bound['l1'] <= j < bound['r1'] and not (bound['t2'] <= i < bound['b2'] and bound['l2'] <= j < bound['r2']):
#                 frame[i][j] = color

def load_pattern(fname, bound):
#     return cv2.imread(fname + ".jpg")
    l2, r2, t2, b2 = unpack(bound, 'l2', 'r2', 't2', 'b2')
    with open(fname + ".txt") as f:
        zero_pixel = json.loads(f.read())
        a = f.read()
        f.close()
        return zero_pixel
    return np.zeros((b2 - t2, r2 - l2, 3))

def save_pattern(fname, frame, bound):
    l2, r2, t2, b2 = unpack(bound, 'l2', 'r2', 't2', 'b2')
    cv2.imwrite(fname + ".jpg", frame[t2:b2,l2:r2])
    with open(fname + ".txt", "w") as f:
        f.write(str(frame[t2:b2,l2:r2].tolist()))
        f.close()

def compare_pixel(frame, last_frame, bound=SCORE, fixed=False):
    l2, r2, t2, b2 = unpack(bound, 'l2', 'r2', 't2', 'b2')
    pixel_mean = abs((last_frame if fixed else last_frame[t2:b2, l2:r2]) - frame[t2:b2, l2:r2]).mean()
    return pixel_mean

def action2tuple(action):
    if action == 0:
        return ("", "")
    elif action == 1:
        return ("w", "")
    elif action == 2:
        return ("", "a")
    elif action == 3:
        return ("s", "")
    else:
        return ("", "d")

def perspectiveTransform(v, M):
    p = np.array((v[0], v[1], 1.))
    p = M.dot(p)
    p = p * (1.0 / p[2])
    return p[0:2]

class Volleyball:
    def __init__(self, serial=True):
        try:
            global testFPS, timer
            testFPS = TestFPS()
            timer = TimeCount()
            global countFPS, arduino_serial
            countFPS = TestFPS()
            arduino_serial = Serial() if serial else Serial_null()
            vc = int(input().split(' ')[0])
        
            cap = cv2.VideoCapture(vc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, Width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # cap.set(cv2.CAP_PROP_FPS, 15)
            # print("FPS:{}".format(cap.get(cv2.CAP_PROP_FPS)))

            print("begin capturing")
            ret, frame = cap.read()
            print("size:", len(frame), len(frame[0]))
            zeroPixel = load_pattern(SCORE_ZERO_FILE, SCORE)
            missPixel = load_pattern(MISS_FILE, MISS)
            gameState = GameState.HANG
            gameScore = 0
            recordPos = [0, 0, 0, 0]
            center = None
            toclass(self, locals(), \
                'cap', 'frame', 'zeroPixel', 'missPixel', 'gameState', 'gameScore', 'center', 'recordPos')
        except:
            print("ERROR when opening the virtual camera")
            raise
        
    def step(self, action=0):
        frame, gameState, gameScore, center, recordPos = \
            self.frame, self.gameState, self.gameScore, self.center, self.recordPos
        try:
          # -- print FPS --
            # global testFPS
            # testFPS.plus()
            # testFPS.print_per_sec()
            # global timer
            # timer.show_per_sec()
            global countFPS, arduino_serial
            countFPS.plus()
            reward = 0

#           -- Capture frame-by-frame --
            # timer.start('getFrame')
            last_frame = frame.copy()
            ret, frame = self.cap.read()
            newFrame = frame.copy()
            # timer.check('getFrame')
            # timer.start('cvSquare')
            # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # thresh = cv2.inRange(hsv, lower_green, upper_green)
            # image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            # # cv2.polylines(frame, (fieldA, fieldB, fieldC), True, (0,0,0), 2)
            # if len(contours) > 0:
            #     c = max(contours, key = cv2.contourArea)
            #     # find the center of the bounding box
            #     x,y,w,h = cv2.boundingRect(c)
            #     last_center = center
            #     center = (int(x+w/2), int(y+h/2+mario_h))
            #     # if last_center != None and abs(np.array(last_center) - np.array(center)).mean() > 10:
            #     #     center = last_center
            #     cv2.circle(frame, center, 2, (255,255,0), 2)
            #     if action == 1 and cv2.pointPolygonTest(fieldA, center, False) >= 0:
            #         action = 0
            #         print("forbid A")
            #     elif action == 3 and cv2.pointPolygonTest(fieldB, center, False) >= 0:
            #         action = 0
            #         print("forbid B")
            #     elif action == 4 and cv2.pointPolygonTest(fieldC, center, False) >= 0:
            #         action = 0
            #         print("forbid C")
            # timer.check('cvSquare');
            # timer.start('stateMachine');

#           -- give a blue bound of score image --
#             draw_frame(frame, SCORE)
#             draw_frame(frame, MISS)

#           -- state machine --
            if gameState == GameState.HANG:
                arduino_serial.qsend(*action2tuple(action))
            elif gameState == GameState.LEAVE:
                # arduino_serial.qsend("d", "")
                if countFPS.frames() < recordPos[0]:
                    arduino_serial.qsend("w", "")
                elif countFPS.frames() < recordPos[1]:
                    arduino_serial.qsend("a", "")
                elif countFPS.frames() < recordPos[2]:
                    arduino_serial.qsend("s", "")
                elif countFPS.frames() < recordPos[3]:
                    arduino_serial.qsend("d", "")
                elif countFPS.frames() < recordPos[3] + 20:
                    pass
                elif countFPS.frames() < recordPos[3] + 60:
                    arduino_serial.qsend("d", "")
                elif countFPS.frames() < recordPos[3] + 120:
                    pass
                else:
                    gameState = GameState.ENTER
                    countFPS.save()

            elif gameState == GameState.ENTER:
                center = None
                if countFPS.frames() < 45:
                    arduino_serial.qsend("a", "")
                    # countFPS.save()
                if 45 <= countFPS.frames() < 50:
                    arduino_serial.qsend("s", "")
                if countFPS.frames() > 55:
                    gameState = GameState.WAIT
                    countFPS.save()
            elif gameState in (GameState.WAIT, GameState.LIVE):
                missed = False
                if gameState == GameState.WAIT:
                    if compare_pixel(frame, self.zeroPixel, bound=SCORE, fixed=True) < 1.:
                        gameState = GameState.LIVE
                        gameScore = 0
                        print("begin zero at frame: %d" % countFPS.frames())
                        countFPS.save()
                    if countFPS.frames() > 140:
                        gameState = GameState.OVER
                        print("bug place")
                        missed = True
                        countFPS.save()
                else:
                    lst = compare_pixel(frame, last_frame, bound=SCORE, fixed=False)
                    if 40. < lst \
                    or compare_pixel(frame, self.missPixel, bound=MISS, fixed=True) < 30.:
                        # reward -= 5
                        print("You get %d points this time! miss frames: %d" % (gameScore, countFPS.frames()))
                        gameState = GameState.OVER
                        missed = True
                        countFPS.save()
                    elif 1.< lst < 40.:
                        gameScore += 1
                        reward += gameScore * 10
                        print("score: %d, total frames: %d" % (gameScore, countFPS.frames()))
                        countFPS.save()

                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                thresh = cv2.inRange(hsv, lower_green, upper_green)
                image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                # cv2.polylines(frame, (fieldA, fieldB, fieldC), True, (0,0,0), 2)
                if len(contours) > 0:
                    c = max(contours, key = cv2.contourArea)
                    # find the center of the bounding box
                    x,y,w,h = cv2.boundingRect(c)
                    center = perspectiveTransform((int(x+w/2), int(y+h/2+mario_h)), M)

                    # if last_center != None and abs(np.array(last_center) - np.array(center)).mean() > 10:
                    #     center = last_center
                    a = center - CENTER
                    reward += 0.1 / 2000 * (MLEN - np.linalg.norm(a))
                    center = tuple(np.int8(center).tolist())
                    cv2.circle(newFrame, (int(x+w/2), int(y+h/2+mario_h)), 2, (255,255,0), 2)
                    cv2.circle(newFrame, (center[0], center[1]), 2, (0,255,0), 2)
                    cv2.circle(newFrame, (CENTER[0], CENTER[1]), 2, (0,0,255), 2)
                    # if action == 1 and cv2.pointPolygonTest(fieldA, center, False) >= 0:
                    #     action = 0
                    #     # print("forbid A")
                    # elif action == 3 and cv2.pointPolygonTest(fieldB, center, False) >= 0:
                    #     action = 0
                    #     # print("forbid B")
                    # elif action == 4 and cv2.pointPolygonTest(fieldC, center, False) >= 0:
                    #     action = 0
                    #     # print("forbid C")
                    if center[0] < boundField[0]:
                        action = 0 if action == 2 else action
                        # print("boundA")
                    if center[0] > boundField[1]:
                        action = 0 if action == 4 else action
                        # print("boundB")
                    if center[1] < boundField[2]:
                        action = 0 if action == 1 else action
                        # print("boundC")
                    if center[1] > boundField[3]:
                        action = 0 if action == 3 else action
                        # print("boundD")

                    if missed:
                        recordPos = [0, 0, 0, 0]
                        if a[0] < 0:
                            recordPos[3] = ((-a[0])**0.5)*4
                        else:
                            recordPos[1] = (a[0]**0.5)*4
                        if a[1] < 0:
                            recordPos[2] = ((-a[1])**0.5)*4
                        else:
                            recordPos[0] = (a[1]**0.5)*4
                        for i in range(1, len(recordPos)):
                            recordPos[i] += recordPos[i-1]
                arduino_serial.qsend(*action2tuple(action))
            elif gameState == GameState.OVER:
                if countFPS.frames() > 30:
                    gameState = GameState.LEAVE
                    countFPS.save()
            # timer.check('stateMachine')
            # timer.start('showNkey')

#           -- Our operations on the frame come here --
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             full = cv2.cvtColor(frame, 0)

#           -- Display the resulting frame --
            
            # dst = cv2.warpPerspective(frame,M,(77,100))
            # for pi in field:
                # print(perspectiveTransform(pi, M))
            cv2.imshow('frame', newFrame)
            inputC = cv2.waitKey(1)
            if inputC == ord('q'):
#                 -- use to capture the 0 score image --
#                 save_pattern(SCORE_ZERO_FILE, frame, SCORE)
#                 save_pattern(MISS_FILE, frame, MISS)
                return (False, None, 0, frame)
            elif inputC == ord('f'):
                if gameState == GameState.HANG:
                    countFPS.save()
                    gameState = GameState.ENTER
                else:
                    gameState = GameState.HANG
            elif inputC == ord('c'):
                print(gameState)
            # timer.check('showNkey')
        except:
            import sys, os
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("unexpected error: ", exc_type, fname, exc_tb.tb_lineno)
            return (False, None, 0, None)
        # timer.start('send')
        toclass(self, locals(), \
            'frame', 'gameState', 'gameScore', 'center', 'recordPos')
        # timer.check('send')
        return (True, gameState, reward, frame)
    
    def wait_for_start(self):
        try:
            while True:
                oneFrame = self.step()
                if not oneFrame[0]:
                    return False
                if oneFrame[1] == GameState.WAIT:
                    return True
        except:
            import sys, os
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print("unexpected error: ", exc_type, fname, exc_tb.tb_lineno)
                
    
    def close(self):
#       -- When everything done, release the capture --
        self.cap.release()
        cv2.destroyAllWindows()
        arduino_serial.close()

if __name__ == "__main__":
    game = Volleyball(False)
    while game.step()[0]:
        pass
    game.close()