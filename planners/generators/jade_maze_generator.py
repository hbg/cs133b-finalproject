import numpy as np
import matplotlib.pyplot as plt

def get_walls(x, y, width, height, visited):
    neighbors = []
    if x >= 0:
        neighbors.append((x - 1, y))
    if y >= 0:
        neighbors.append((x, y - 1))
    if x <= width - 1:
        neighbors.append((x + 1, y))
    if y <= height - 1:
        neighbors.append((x, y + 1))
    return neighbors

def generate_maze(width, height, path_probability=0.1):  # Adjust path_probability for clearer paths
    grid = np.ones(shape=(height, width))
    walls = []
    visited = set()
    start_x = np.random.randint(0, width)
    start_y = np.random.randint(0, height)
    visited.add((start_x, start_y))
    walls.extend(get_walls(start_x, start_y, width, height, visited))
    while walls:
        i = np.random.randint(0, len(walls))
        wall = walls[i]
        x, y = wall
        neighbors = get_walls(x, y, width, height, visited)
        visited_neighbors = []
        for neighbor in neighbors:
            if neighbor in visited:
                visited_neighbors.append(neighbor)
        if len(visited_neighbors) == 1 or np.random.rand() < path_probability:
            grid[y, x] = 0
            visited.add((x, y))
            walls.extend(neighbors)
        walls.remove(wall)
    grid_with_edges = np.zeros(shape=(height + 2, width + 2))
    grid_with_edges[1:-1, 1:-1] = grid
    return grid_with_edges

maze = generate_maze(50, 50, path_probability=0.0001)
plt.pcolormesh(maze, cmap='binary')
plt.show()
