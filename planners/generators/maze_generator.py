
import numpy as np
import matplotlib.pyplot as plt
import random

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

def generate_maze(width, height):
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
    return grid

if __name__ == "__main__":
    maze = generate_maze(51, 51)
    plt.pcolormesh(maze, cmap='binary')
    plt.show()