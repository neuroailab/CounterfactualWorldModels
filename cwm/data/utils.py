import os
import numpy as np
import copy
from PIL import Image
import torch
from torchvision import transforms
import kornia
import cv2
import torch
import torch.nn as nn

class RgbFlowToXY(object):
    def __init__(self, to_image_coordinates=True, to_sampling_grid=False):
        self.to_image_coordinates = to_image_coordinates
        self.to_sampling_grid = to_sampling_grid
    def __call__(self, flows_rgb):
        return rgb_to_xy_flows(flows_rgb, self.to_image_coordinates, self.to_sampling_grid)

class FlowToRgb(object):

    def __init__(self, max_speed=1.0, from_image_coordinates=False, from_sampling_grid=True):
        self.max_speed = max_speed
        self.from_image_coordinates = from_image_coordinates
        self.from_sampling_grid = from_sampling_grid

    def __call__(self, flow):
        assert flow.size(-3) == 2, flow.shape
        if self.from_sampling_grid:
            flow_x, flow_y = torch.split(flow, [1, 1], dim=-3)
            flow_y = -flow_y
        elif not self.from_image_coordinates:
            flow_x, flow_y = torch.split(flow, [1, 1], dim=-3)
        else:
            flow_h, flow_w = torch.split(flow, [1,1], dim=-3)
            flow_x, flow_y = [flow_w, -flow_h]

        angle = torch.atan2(flow_y, flow_x) # in radians from -pi to pi
        speed = torch.sqrt(flow_x**2 + flow_y**2) / self.max_speed

        hue = torch.fmod(angle, torch.tensor(2 * np.pi))
        sat = torch.ones_like(hue)
        val = speed

        hsv = torch.cat([hue, sat, val], -3)
        rgb = kornia.color.hsv_to_rgb(hsv)
        return rgb

class OpticalFlowRgbTo2d(object):

    def __init__(self, channels_last=False, max_speed=1.0, to_image_coordinates=True):
        self.channels_last = channels_last
        self.max_speed = max_speed
        self.to_image_coordinates = to_image_coordinates

    @staticmethod
    def hsv_to_2d_velocities_and_speed(hsv, max_speed=1.0, to_image_coordinates=False):
        if hsv.dtype == np.uint8:
            hsv = hsv / 255.0
        h,s,v = hsv[...,0], hsv[...,1], hsv[...,2]
        ang = h * 2 * np.pi
        speed = v * max_speed
        flow_x = np.cos(ang) * speed
        flow_y = np.sin(ang) * speed
        mag = np.sqrt(flow_x**2 + flow_y**2)

        if to_image_coordinates:
            flow = np.stack([-flow_y, flow_x, mag], -1)
        else:
            flow = np.stack([flow_x, flow_y, mag], -1)

        return flow

    def __call__(self, rgb_flows):

        if rgb_flows.dtype == np.float32:
            rgb_flows = np.clip(rgb_flows * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            assert rgb_flows.dtype == np.uint8

        H,W,C = rgb_flows.shape
        out = np.zeros((H,W,3), dtype=np.float32)
        hsv = cv2.cvtColor(rgb_flows, cv2.COLOR_RGB2HSV)
        velocities = self.hsv_to_2d_velocities_and_speed(
            hsv,
            max_speed=self.max_speed,
            to_image_coordinates=self.to_image_coordinates
        )
        out = velocities
        return out
