import numpy as np
import re


class Coordinates:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Figure:
    def __init__(self, points: np.ndarray):
        self.points = points

    def get_points(self) -> np.ndarray:
        return self.points


def read_config(path: str) -> np.ndarray:
    file = open(path, "r")
    num_figures = int(file.readline())
    figures = np.ndarray(shape=(num_figures,), dtype=Figure)

    for i in range(num_figures):
        string = file.readline()
        coordinates = re.findall(r'\d+', string)
        assert len(coordinates) % 2 == 0
        coordinates = np.array(coordinates).reshape(len(coordinates) // 2, 2)
        points = np.ndarray(shape=(coordinates.shape[0]), dtype=Coordinates)
        for j in range(coordinates.shape[0]):
            points[j] = Coordinates(coordinates[j][0], coordinates[j][1])
        figures[i] = Figure(points)

    return figures
