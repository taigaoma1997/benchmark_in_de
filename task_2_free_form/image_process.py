# -*- coding: utf-8 -*-
# @Author: Brandon Han
# @Date:   2019-09-03 11:44:56
# @Last Modified by:   Brandon Han
# @Last Modified time: 2019-09-06 00:24:53

import cv2
import numpy as np
import random
import math


def clip(x, min, max):
    if(min > max):
        return x
    elif(x < min):
        return min
    elif(x > max):
        return max
    else:
        return x


def generate_verts(ctrX, ctrY, aveRadius, irregularity, spikeyness, numVerts):

    irregularity = clip(irregularity, 0, 1) * 2 * math.pi / numVerts
    spikeyness = clip(spikeyness, 0, 1) * aveRadius

    # generate n angle steps
    angleSteps = []
    lower = (2 * math.pi / numVerts) - irregularity
    upper = (2 * math.pi / numVerts) + irregularity
    sum = 0
    for i in range(numVerts):
        tmp = random.uniform(lower, upper)
        angleSteps.append(tmp)
        sum = sum + tmp

    # normalize the steps so that point 0 and point n+1 are the same
    k = sum / (2 * math.pi)
    for i in range(numVerts):
        angleSteps[i] = angleSteps[i] / k

    # now generate the points
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(numVerts):
        r_i = clip(random.gauss(aveRadius, spikeyness), 0, 2 * aveRadius)
        x = ctrX + r_i * math.cos(angle)
        y = ctrY + r_i * math.sin(angle)
        points.append((int(x), int(y)))

        angle = angle + angleSteps[i]

    return points


class MetaShape(object):
    """docstring for MetaShape"""

    def __init__(self, width):
        super(MetaShape, self).__init__()
        self.width = width
        self.ctrX = width / 2
        self.ctrY = width / 2
        self.img = np.zeros((width, width), dtype=np.uint8)

    def init_verts(self):
        self.aveRadius = random.uniform(width / 5, width / 3)
        self.irregularity = random.uniform(0, 0.9)
        self.spikeyness = random.uniform(0, 0.4)
        self.numVerts = random.randint(3, 20)
        self.verts = generate_verts(self.ctrX, self.ctrY, self.aveRadius,
                                    self.irregularity, self.spikeyness, self.numVerts)

    def draw_polygon(self):
        for i in range(self.numVerts - 1):
            cv2.line(self.img, self.verts[i], self.verts[i + 1], color=255, thickness=1)
        cv2.line(self.img, self.verts[self.numVerts - 1], self.verts[0], color=255, thickness=1)

    def fill_polygon(self):
        temp_img = self.img.copy()
        mask = np.zeros((self.width + 2, self.width + 2), np.uint8)
        cv2.floodFill(temp_img, mask, (0, 0), 255)
        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(temp_img)
        # Combine the two images to get the foreground.
        self.img = self.img | im_floodfill_inv

    def blur_polygon(self, kernel_size=15):
        self.img = cv2.GaussianBlur(self.img, (kernel_size, kernel_size), 0)

    def resize_polygon(self, new_size=64):
        self.img = cv2.resize(self.img, (new_size, new_size))

    def binary_polygon(self):
        _, self.img = cv2.threshold(self.img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def pad_boundary(self):
        pad_pixel = int(np.ceil(20 / self.width * 64))
        new_img = np.zeros_like(self.img)
        new_img[pad_pixel:64 - pad_pixel, pad_pixel:64 -
                pad_pixel] = self.img[pad_pixel:64 - pad_pixel, pad_pixel:64 - pad_pixel]
        self.img = new_img

    def erode_dilate(self, struc=5, iterations=3, mode="close"):
        kernel = np.ones((struc, struc), np.uint8)
        if mode == "close":
            self.img = cv2.dilate(self.img, kernel, iterations=iterations)
            self.img = cv2.erode(self.img, kernel, iterations=iterations)
        elif mode == "open":
            self.img = cv2.erode(self.img, kernel, iterations=iterations)
            self.img = cv2.dilate(self.img, kernel, iterations=iterations)
        elif mode == "erode":
            self.img = cv2.erode(self.img, kernel, iterations=iterations)

    def remove_small_twice(self, ratio=0.1):

        def remove_small(img, ratio):
            threshold = np.floor(ratio * np.sum(img) / 255)
            # find all your connected components (white blobs in your image)
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
            sizes = stats[1:, -1]
            nb_components = nb_components - 1
            img2 = np.zeros(output.shape, dtype=np.uint8)
            # for every component in the image, you keep it only if it's above threshold
            for i in range(0, nb_components):
                if sizes[i] >= threshold:
                    img2[output == i + 1] = 255
            return img2

        self.img = remove_small(self.img, ratio)
        self.img = remove_small(255 - self.img, ratio)
        self.img = 255 - self.img

    def save_polygon(self, name):
        cv2.imwrite(name, self.img)

    def show_polygon(self, time=2000):
        cv2.imshow("MetaShape", self.img)
        cv2.waitKey(time)

    def check_size(self):
        if np.sum(self.img) < self.width ** 2:
            return False
        else:
            return True

    def generate_polygon(self, name):
        self.init_verts()
        self.draw_polygon()
        self.fill_polygon()
        self.blur_polygon()
        self.resize_polygon()
        self.pad_boundary()
        self.binary_polygon()
        if self.check_size():
            self.save_polygon(name)
            # self.show_polygon()


if __name__ == '__main__':
    times = 10
    begin = 0
    for i in range(begin, begin + times):
        #width = random.randint(200, 400)
        width = 340
        polyogn = MetaShape(width)
        polyogn.generate_polygon('test/' + str(i) + '_' + str(width) + '.png')
