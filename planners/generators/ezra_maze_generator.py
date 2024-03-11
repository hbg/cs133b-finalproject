
import random

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.prepared import prep

EMPTY = 0
WALL = 1
START = 2
END = 3
KEY = 4
LOCK = 5

keys = []
locks = []
key_count = 10
lock_count = 25

def get_neighbor_passages(x, y, width, height, grid):
    neighbors = []
    if x - 2 > 0 and grid[y, x - 2] == 0:
        neighbors.append((x - 2, y))
    if y - 2 > 0 and grid[y - 2, x] == 0:
        neighbors.append((x, y - 2))
    if x + 2 < width - 1 and grid[y, x + 2] == 0:
        neighbors.append((x + 2, y))
    if y + 2 < height - 1 and grid[y + 2, x] == 0:
        neighbors.append((x, y + 2))
    return neighbors

def get_neighbor_walls(x, y, width, height, grid):
    neighbors = []
    if x - 2 > 0 and grid[y, x - 2] == 1:
        neighbors.append((x - 2, y))
    if y - 2 > 0 and grid[y - 2, x] == 1:
        neighbors.append((x, y - 2))
    if x + 2 < width - 1 and grid[y, x + 2] == 1:
        neighbors.append((x + 2, y))
    if y + 2 < height - 1 and grid[y + 2, x] == 1:
        neighbors.append((x, y + 2))
    return neighbors

def remove_key_and_lock(grid, index):
    key = keys[index]
    lock = locks[index]
    grid[key[1]][key[0]] = 0
    grid[lock[1]][lock[0]] = 0

def is_solvable(grid, queue, width, height):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = set()

    while len(queue) != 0:
        start_x, start_y = queue.pop(0)
        if grid[start_y][start_x] == END:
            return True
        
        visited.add((start_x, start_y))
        for dx, dy in directions:
            new_x, new_y = start_x + dx, start_y + dy
            if new_x >= 0 and new_x < width and new_y >= 0 and new_y < height and (new_x, new_y) not in visited:
                cell = grid[new_y][new_x]
                if cell == END:
                    return True
                elif cell == KEY:
                    remove_key_and_lock(grid, keys.index((new_x, new_y)))
                    queue.append((new_x, new_y))
                elif cell == EMPTY:
                    queue.append((new_x, new_y))
                elif cell == LOCK:
                    all_locks = True
                    for x, y in queue:
                        if grid[y][x] != LOCK:
                            queue.append((new_x, new_y))
                            all_locks = False
                    if all_locks:
                        return False
    return False

def generate_maze(width, height, num_keys):
    grid = np.ones(shape=(height, width))
    start_x = np.random.randint(0, width - 1)
    start_y = np.random.randint(0, height - 1)
    grid[start_y, start_x] = 0
    visited = {(start_x, start_y)}
    walls = get_neighbor_walls(start_x, start_y, width, height, grid)
    while len(walls) > 0:
        x, y = random.choice(walls)
        if (x, y) not in visited:
            grid[y, x] = 0
            neighbors = get_neighbor_passages(x, y, width, height, grid)
            if len(neighbors) > 0:
                random_neighbor = random.choice(neighbors)
                new_passage = (
                    x + (random_neighbor[0] - x) // 2,
                    y + (random_neighbor[1] - y) // 2
                )
                grid[new_passage[1], new_passage[0]] = 0
                visited.add(new_passage)
                new_walls = get_neighbor_walls(x, y, width, height, grid)
                walls.remove((x, y))
                walls.extend(new_walls)
            else:
                walls.remove((x, y))
            visited.add((x, y))
        else:
            walls.remove((x, y))

    start_x = np.random.randint(0, width - 1)
    start_y = np.random.randint(0, height - 1)
    while grid[start_y, start_x] != 0:
        start_x = np.random.randint(0, width - 1)
        start_y = np.random.randint(0, height - 1)
    grid[start_y, start_x] = START

    end_x = np.random.randint(0, width - 1)
    end_y = np.random.randint(0, height - 1)
    while grid[end_y, end_x] != 0:
        end_x = np.random.randint(0, width - 1)
        end_y = np.random.randint(0, height - 1)
    grid[end_y, end_x] = END

    for _ in range(num_keys):
        key_x = np.random.randint(0, width - 1)
        key_y = np.random.randint(0, height - 1)
        while grid[key_y, key_x] != 0:
            key_x = np.random.randint(0, width - 1)
            key_y = np.random.randint(0, height - 1)
        grid[key_y, key_x] = KEY
        keys.append((key_x, key_y))

        lock_x = np.random.randint(0, width - 1)
        lock_y = np.random.randint(0, height - 1)
        while grid[lock_y, lock_x] != 0:
            lock_x = np.random.randint(0, width - 1)
            lock_y = np.random.randint(0, height - 1)
        grid[lock_y, lock_x] = LOCK
        locks.append((lock_x, lock_y))
    
    if is_solvable(grid.copy(), [(start_x, start_y)], width, height):
        return (grid, keys, locks, (start_x, start_y), (end_x, end_y))
    else:
        print('generating new maze.')
        return generate_maze(width, height, num_keys)

def generate_maze_polygons(width, height, difficulty):
    polys = []
    maze = generate_maze(width, height)
    for i in range(width):
        for j in range(height):
            x_border_dist = min(i, width - i)
            y_border_dist = min(j, height - j)
            if maze[j, i] > 0:
                if (x_border_dist <= 2 or y_border_dist <= 2) or \
                      random.random() < difficulty:
                    if (i - 1 >= 0) and maze[j, i - 1] > 0:
                        polys.append(Polygon([[i, j+0.45], [i+0.5, j+0.4],
                                              [i+0.5, j+0.55], [i, j+0.55]]))
                    if (i + 1 < width) and maze[j, i + 1] > 0:
                        polys.append(Polygon([[i + 0.5, j+0.45], [i+1, j+0.45],
                                              [i+1, j+0.55], [i+0.5, j+0.55]]))
                    if (j - 1 >= 0) and maze[j - 1, i] > 0:
                        polys.append(Polygon([[i+0.45, j], [i+0.45, j+0.5],
                                              [i+0.55,j+0.5], [i+0.55,j]]))
                    if (j + 1 < height) and maze[j + 1, i] > 0:
                        polys.append(Polygon([[i+0.45, j+0.5], [i+0.45, j+1],
                                              [i+0.55,j+1], [i+0.55,j+0.5]]))
    filled_grids = prep(MultiPolygon(polys))
    return filled_grids

if __name__ == "__main__":
    maze = generate_maze(51, 51)
    plt.pcolormesh(maze, cmap='binary')
    plt.show()