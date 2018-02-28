import numpy as np
import cv2
import math
from VolleyBall import Width, Height

field = np.array([[190, 145], [190, 290], [435, 290], [390, 145]])

def make_mask(width, height, contour):
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [contour], (255)) 
    return mask

mask_field = make_mask(Width, Height,field)

# upper and lower range of the green color of the luigi hat
CLR_HAT = ((45,150,40), (57,210,220))
# upper and lower range of the shadow color
CLR_SHD = ((14,127,153), (15,153,178))
# upper and lower range of the ball color
CLR_BALL = ((175,150,0), (5,250,220))

CLR_RNGS = [CLR_HAT, CLR_SHD, CLR_BALL]
masks = [None, mask_field, None]
# masks = [None, None, None]

class ObjectsTracker():
    """Tracks objects in a video"""
    def __init__(self, object_num, color_ranges, masks, maxDist = 40, drawCntrs = False):
        """color_ranges and masks must be the same length with object_num"""
        """maxDist is the largest distance an object can move between frames"""
        assert len(color_ranges) == object_num
        assert len(masks) == object_num
        self.last_pos = [None]*object_num
        self.color_ranges = color_ranges
        self.masks = masks
        self.maxDist = maxDist
        self.drawCntrs = drawCntrs

    def detect_color(hsv_img, lower_range, upper_range, mask=None, minArea = 4):
        if mask is not None:
            hsv_img = cv2.bitwise_and(hsv_img, hsv_img, mask = mask)
        if lower_range[0] > upper_range[0]:
            thresh1 = cv2.inRange(hsv_img, (0,lower_range[1],lower_range[2]), upper_range)
            thresh2 = cv2.inRange(hsv_img, lower_range, (180, upper_range[1], upper_range[2]))
            thresh = cv2.bitwise_or(thresh1, thresh2)
        else:
            thresh = cv2.inRange(hsv_img, lower_range, upper_range)
        image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            # contour not detected 
            return None
        # the largest area is probably the one
        cnt = max(contours, key = cv2.contourArea)
        if cv2.contourArea(cnt) > minArea:

            return cnt
        else:
            return None

    def update(self, frame):
        contours = []
        for i, color_range in enumerate(self.color_ranges):
            lr, ur = color_range
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            contours.append(ObjectsTracker.detect_color(hsv, lr, ur, self.masks[i], minArea = 6))

            
        centers = []
        for i, cnt in enumerate(contours):
            pos = None
            if cnt is not None:
                if self.drawCntrs:
                    cv2.drawContours(frame, [cnt], 0, (255,255,0), 3)
                M = cv2.moments(cnt)
                pos = np.array([int(M['m10']/M['m00']), int(M['m01']/M['m00'])])
                lpos = self.last_pos[i]
                if lpos is not None and np.linalg.norm(lpos-pos) > self.maxDist:
                    pos = None
                    centers.append(lpos)
                else:
                    centers.append(pos)
            else:
                centers.append(None)
            self.last_pos[i] = pos
        return centers, frame

objTracker = ObjectsTracker(3, CLR_RNGS, masks)