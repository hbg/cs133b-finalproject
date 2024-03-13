from generators.maze_generator import generate_maze_with_keys, maze_to_polygons
from shapely.geometry import MultiPolygon, Polygon
from shapely.prepared import prep
import numpy as np

class Maze:
    def __init__(self, width, height, num_keys, difficulty=1.0):
        self.width = width
        self.height = height
        self.num_keys = num_keys
        self.maze, self.keys, self.locks, self.start, self.goal = generate_maze_with_keys(width, height, num_keys)
        self.wall_polys = maze_to_polygons(self.maze, difficulty)
        self.lock_polys = []
        for (i, j) in self.locks:
            self.lock_polys.append(Polygon([[i - 1, j - 1], [i + 1, j - 1], [i+1, j+1], [i - 1, j+1]]))
        self.lock_polys = MultiPolygon(self.lock_polys)
        self.wall_polys_prep = prep(self.wall_polys)
        self.lock_polys_prep = prep(self.lock_polys)

    def disjoint(self, polygon: Polygon):
        return self.wall_polys_prep.disjoint(polygon) and self.lock_polys_prep.disjoint(polygon)

    def set_locks(self, locks):
        self.locks = locks

    def set_lock_polys(self, lock_polys):
        self.lock_polys = lock_polys
        self.lock_polys_prep = prep(lock_polys)

    def get_keys(self):
        return self.keys

    def get_start(self):
        return self.start

    def get_goal(self):
        return self.goal