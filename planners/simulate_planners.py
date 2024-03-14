import est_planner
import est_90_planner
import ezra_cardinal_rrt_planner
from generators.maze import Maze
import matplotlib.pyplot as plt
import rrt_90_planner
import rrt_planner
import numpy as np

iterations = 5
num_mazes = 1
DIFFICULTIES = [.75, 1]
NUM_KEYS = [0, 5, 10]
MAZE_SIZES = [21, 41]

plt.figure(figsize=(12, 8))

for difficulty in DIFFICULTIES:
    for num_keys in NUM_KEYS:
        for size in MAZE_SIZES:
            means = {'EST': [], 'EST 90': [], 'RRT': [], 'RRT 90': []}
            for i in range(num_mazes):
                np.random.seed(i)
                maze = Maze(size, size, num_keys, difficulty)
                planners = ['EST', 'EST 90']

                # Simulate EST
                est_data = []
                for j in range(iterations):
                    nodes = est_planner.main(seed_maze=maze, visual=False)
                    if nodes == -1:
                        est_data.append(1500)
                    else:
                        est_data.append(nodes)
                means['EST'].append(np.mean(est_data))

                # Simulate EST 90
                est_90_data = []
                for j in range(iterations):
                    nodes = est_90_planner.main(seed_maze=maze, visual=False)
                    if nodes == -1:
                        est_90_data.append(1500)
                    else:
                        est_90_data.append(nodes)
                means['EST 90'].append(np.mean(est_90_data))

                # Simulate RRT
                rrt_data = []
                for j in range(iterations):
                    nodes = rrt_planner.main(seed_maze=maze, visual=False)
                    if nodes == -1:
                        rrt_data.append(1500)
                    else:
                        rrt_data.append(nodes)
                means['RRT'].append(np.mean(rrt_data))

                # Simulate RRT 90
                rrt_90_data = []
                for j in range(iterations):
                    nodes = rrt_90_planner.main(seed_maze=maze, visual=False)
                    if nodes == -1:
                        rrt_90_data.append(1500)
                    else:
                        rrt_90_data.append(nodes)
                means['RRT 90'].append(np.mean(rrt_90_data))
                
                plt.subplot(len(DIFFICULTIES), len(NUM_KEYS), (DIFFICULTIES.index(difficulty) * len(NUM_KEYS)) + (NUM_KEYS.index(num_keys) + 1))
                plt.bar(means.keys(), [np.mean(means['EST']), np.mean(means['EST 90']), np.mean(means['RRT']), np.mean(means['RRT 90'])], width=0.4)
                plt.title(f"Maze Size: {size}, Num Keys: {num_keys}, Difficulty: {difficulty}")
                plt.xlabel('Planners')
                plt.ylabel('Mean Nodes Expanded')
plt.show()
    
