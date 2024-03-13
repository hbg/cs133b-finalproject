import est_planner
import ezra_cardinal_rrt_planner
from generators.maze import Maze
import matplotlib.pyplot as plt
import rrt_90_planner
import rrt_planner
import numpy as np

iterations = 50
num_mazes = 1
difficulty = 1
num_keys = 10
WIDTH = 41
HEIGHT = 41

for i in range(num_mazes):
    np.random.seed(i)
    maze = Maze(WIDTH, HEIGHT, num_keys, difficulty)

    # Simulate EST
    est_data = []
    for j in range(iterations):
        nodes = est_planner.main(seed_maze=maze, visual=False)
        if nodes == -1:
            est_data.append(0)
        else:
            est_data.append(nodes)
    plt.plot(est_data)
    plt.show()
    
