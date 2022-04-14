import math
from copy import copy
from itertools import groupby

import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

intent = 6


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def get_x(self) -> int: return self.x

    def get_y(self) -> int: return self.y

    def compare(self, other) -> bool: return math.fabs(self.x - other.get_x()) < intent and math.fabs(
        self.y - other.get_y()) < intent


class Line:
    def __init__(self, point1: Point, point2: Point):
        self.point1 = point1
        self.point2 = point2

    def get_point1(self) -> Point: return self.point1

    def get_point2(self) -> Point: return self.point2

    def point1_as_tuple(self) -> tuple: return self.point1.get_x(), self.point1.get_y()

    def point2_as_tuple(self) -> tuple: return self.point2.get_x(), self.point2.get_y()

    def is_continue(self, con) -> bool:
        if con == self: return False
        return self.point1.compare(con.get_point1()) or self.point1.compare(con.get_point2()) or \
               self.point2.compare(con.get_point1()) or self.point2.compare(con.get_point2())

    def is_end_point(self, point: Point):
        return self.point1.compare(point) or self.point2.compare(point)


def find_lines(path: str):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    kernel = np.array([[1, 0],
                       [0, 1]], dtype=np.uint8)
    gray1 = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    kernel = np.array([[0, 1],
                       [1, 0]], dtype=np.uint8)

    gray2 = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    gray = cv2.bitwise_or(gray1, gray2)
    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    plt.imshow(gray)
    plt.show()

    find_lines = []

    lines = cv2.HoughLinesP(gray, 1, np.pi / 360, 10, maxLineGap=11, minLineLength=2)
    cdst = np.zeros([gray.shape[0], gray.shape[1]], dtype=np.uint8)
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            find_lines.append(Line(Point(l[0], l[1]), Point(l[2], l[3])))
            cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (255))

    cdst = cv2.bitwise_xor(gray, cdst)

    lines = cv2.HoughLinesP(cdst, 1, np.pi / 360, 10, maxLineGap=11, minLineLength=5)
    if lines is not None:
        for i in range(len(lines)):
            l = lines[i][0]
            find_lines.append(Line(Point(l[0], l[1]), Point(l[2], l[3])))
            # cv2.line(cdst, (l[0], l[1]), (l[2], l[3]), (255))

    return np.array(find_lines)


def find_cycle(dictionary: dict, index: int, start_index: int) -> list:
    if dictionary.get(index) is None: return []

    for i in dictionary[index]:
        if i == start_index:
            yield [index]

    for i in dictionary[index]:
        new_dict = copy(dictionary)
        for elem in new_dict[index]:
            if new_dict.get(elem) is not None and index in new_dict[elem]:
                new_dict[elem].remove(index)
        new_dict.pop(index)
        for result in find_cycle(new_dict, i, start_index):
            if len(result) > 0:
                yield result + [index]

    return []


def find_figures(lines: np.ndarray) -> np.ndarray:
    dictionary = {}
    for i in range(lines.shape[0]):
        for j in range(lines.shape[0]):
            if i != j and lines[i].is_continue(lines[j]):
                if dictionary.get(i) is None:
                    dictionary[i] = [j]
                else:
                    dictionary[i].append(j)

    result = []
    for key, values in dictionary.items():
        for value in values:
            new_dict = copy(dictionary)
            if key in new_dict[value]:
                new_dict[value].remove(key)
            for figure in find_cycle(new_dict, value, key):
                if len(figure) > 2:
                    result.append(figure)

    sort = []
    for elem in result:
        sort.append(sorted(elem))

    result = [el for el, _ in groupby(sort)]

    return np.array([[e for e in lines[elem]] for elem in result])

def check_figure(lines: list) -> list:
    for line in lines:
        p1 = False
        p2 = False
        copy_lines = copy(lines)
        copy_lines.remove(line)
        for other in copy_lines:
            if other.is_end_point(line.get_point1()): p1 = True
            if other.is_end_point(line.get_point2()): p2 = True
        if not p1 or not p2:
            lines.remove(line)
            return check_figure(lines)
    return lines

def get_figures(path: str) -> np.ndarray:
    lines = find_lines(path)
    image = np.zeros(shape=(200, 300, 3), dtype=np.uint8)
    for line in lines:
        cv2.line(image, line.point1_as_tuple(), line.point2_as_tuple(), (255, 255, 255))

    image = np.zeros(shape=(200, 300, 3), dtype=np.uint8)
    figures = find_figures(lines)
    for figure in figures:
        print(len(figure))
        for line in figure:
            cv2.line(image, line.point1_as_tuple(), line.point2_as_tuple(), (255, 255, 255))

    kernel = np.array([[0, 1],
                      [1, 0]], dtype=np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(image, contours, -1, color=(255, 255, 255), thickness=cv2.FILLED)

    plt.imshow(image)
    plt.show()

    lines = cv2.HoughLinesP(image, 1, np.pi / 360, 10, maxLineGap=11, minLineLength=2)
    image = np.ndarray(image.shape, dtype=np.uint8)

    for index, contour in enumerate(contours):
        cv2.drawContours(image, contours, index, color=(255, 255, 255), thickness=1)

    plt.imshow(image)
    plt.show()

    return np.array(figures)
