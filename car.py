import ultralytics.engine.results
from ultralytics import *

class Car():
    def __init__(self):
        self.box = ultralytics.engine.results.Boxes
        self.id = None
        self.center = None
        self.GL = False #  Green Light
        self.s_point = None #  Starting mask
        self.last_frame = 0
        self.mask_l = {
            '0': 0,
            '1': 0,
            '2': 0,
            '3': 0,
            '4': 0,
            '5': 0,
            '6': 0,
            '7': 0,
            '8': 0,
            '9': 0,
            '10': 0,
            '11': 0,
            '12': 0,
            '13': 0,
            '14': 0,
            '15': 0
        } #  Mask list (all masks in the intersection)

    def set_box(self,box):
        self.box = box

    def set_id(self,id):
        self.id = id

    def set_center(self,center):
        self.center = center

    def is_GL(self, val):
        self.GL = val

    def set_s_point(self,s_point):
        self.s_point = s_point

    def inc_mask_l(self, list):
        for mask in list:
            if str(mask) in self.mask_l:
                self.mask_l[str(mask)] += 1

    def set_last_frame(self, time):
        self.last_frame = time

