#!/usr/bin/env python3
#
#   rrttriangles.py
#
#   Use RRT to find a path around triangular obstacles.
#
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from math               import pi, sin, cos, atan2, sqrt, ceil
from scipy.spatial      import KDTree
from shapely.geometry   import Point, LineString, Polygon, MultiPolygon
from shapely.prepared   import prep
from generators.maze import Maze

######################################################################
#
#   Parameters
#
#   Define the step size.  Also set the maximum number of nodes.
#
DSTEP = 1.5

# Maximum number of steps (attempts) or nodes (successful steps).
SMAX = 500000
NMAX = 1500
tree_size = 0

######################################################################
#
#   World Definitions
#
#   List of obstacles/objects as well as the start/goal.
#
difficulty = 1
num_keys = 10
WIDTH = 41
HEIGHT = 41
(xmin, xmax) = (0, WIDTH)
(ymin, ymax) = (0, HEIGHT)

np.random.seed(2)
maze = Maze(WIDTH, HEIGHT, num_keys, difficulty)


# Define the start/goal states (x, y, theta)


######################################################################
#
#   Utilities: Visualization
#
# Visualization Class
class Visualization:
    def __init__(self):
        # Clear the current, or create a new figure.
        plt.clf()

        # Create a new axes, enable the grid, and set axis limits.
        plt.axes()
        plt.grid(True)
        plt.gca().axis('on')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)
        plt.gca().set_aspect('equal')

        for poly in maze.wall_polys_prep.context.geoms:
            plt.plot(*poly.exterior.xy, 'k-', linewidth=2)

        for poly in maze.lock_polys_prep.context.geoms:
            plt.plot(*poly.exterior.xy, 'c-', linewidth=2)

        # Show.
        self.show()

    def show(self, text = ''):
        # Show the plot.
        plt.pause(0.0001)
        # If text is specified, print and wait for confirmation.
        if len(text)>0:
            input(text + ' (hit return to continue)')

    def drawNode(self, node, *args, **kwargs):
        plt.plot(node.x, node.y, *args, **kwargs)

    def drawEdge(self, head, tail, *args, **kwargs):
        plt.plot((head.x, tail.x),
                 (head.y, tail.y), *args, **kwargs)

    def drawPath(self, path, *args, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], *args, **kwargs)


######################################################################
#
#   Node Definition
#
class Node:
    def __init__(self, x, y):
        # Define a parent (cleared for now).
        self.parent = None

        # Define/remember the state/coordinates (x,y).
        self.x = x
        self.y = y

    ############
    # Utilities:
    # In case we want to print the node.
    def __repr__(self):
        return ("<Point %5.2f,%5.2f>" % (self.x, self.y))

    # Compute/create an intermediate node.  This can be useful if you
    # need to check the local planner by testing intermediate nodes.
    def intermediate(self, other, alpha):
        return Node(self.x + alpha * (other.x - self.x),
                    self.y + alpha * (other.y - self.y))

    # Return a tuple of coordinates, used to compute Euclidean distance.
    def coordinates(self):
        return (self.x, self.y)

    # Compute the relative distance to another node.
    def distance(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    ################
    # Collision functions:
    # Check whether in free space.
    def inFreespace(self):
        if (self.x <= xmin or self.x >= xmax or
            self.y <= ymin or self.y >= ymax):
            return False
        return maze.disjoint(Point(self.coordinates()))

    # Check the local planner - whether this connects to another node.
    def connectsTo(self, other):
        line = LineString([self.coordinates(), other.coordinates()])
        return maze.disjoint(line)


######################################################################
#
#   RRT Functions
#
def rrt(startnode, goalnode, keylist, visual=True):
    global tree_size
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    tree = [startnode]

    # Function to attach a new node to an existing node: attach the
    # parent, add to the tree, and show in the figure.
    def addtotree(oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)
        if visual:
            visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
            visual.show()

    # Loop - keep growing the tree.
    steps = 0
    P = 0.05
    keys_collected = 0
    current_node = startnode
    while True:
        # Determine the target state.
        if np.random.random() <= P:
            targetnode = goalnode
        else:
            if np.random.random() < 0.5:
                targetnode = Node(random.uniform(0, 41), current_node.y)
            else:
                targetnode = Node(current_node.x, random.uniform(0, 41))

        # Directly determine the distances to the target node.
        distances = np.array([node.distance(targetnode) for node in tree])
        index     = np.argmin(distances)
        nearnode  = tree[index]
        current_node = nearnode
        d         = distances[index]

        # Determine the next node.
        alpha = min(1, DSTEP / d)
        nextnode = nearnode.intermediate(targetnode, alpha)

        # Check whether to attach.
        if nextnode.inFreespace() and nearnode.connectsTo(nextnode):
            addtotree(nearnode, nextnode)

            # If within DSTEP, also try connecting to the goal.  If
            # the connection is made, break the loop to stop growing.
            if nextnode.distance(goalnode) < DSTEP and nextnode.connectsTo(goalnode):
                addtotree(nextnode, goalnode)
                break

            # Check if we can grab a key as well
            deleted_indices = set()
            for i in range(len(keylist)):
                if nextnode.distance(keylist[i]) < DSTEP and nextnode.connectsTo(keylist[i]):
                    deleted_indices.add(i)
                    addtotree(nextnode, keylist[i])
                    keys_collected += 1
                    if visual:
                        visual.show()
                        visual.drawNode(keylist[i], color='red', marker='o') # change key color when collected
                    print("Key collected!")
            if visual:
                lock_polys = MultiPolygon([poly for idx, poly in enumerate(maze.lock_polys.geoms) if idx not in deleted_indices])
                unlocked_poly = MultiPolygon([poly for idx, poly in enumerate(maze.lock_polys.geoms) if idx in deleted_indices])
                maze.set_lock_polys(lock_polys)

                unlocked_poly_prep = prep(unlocked_poly)
                for unlock_poly in unlocked_poly_prep.context.geoms:
                    plt.plot(*unlock_poly.exterior.xy, color='red', linewidth=2)

            new_key_list = []
            for i, elem in enumerate(keylist):
                if i not in deleted_indices:
                    new_key_list.append(elem)
            keylist = new_key_list

        # Check whether we should abort - too many steps or nodes.
        steps += 1
        if (steps >= SMAX) or (len(tree) >= NMAX):
            tree_size = len(tree)
            print("Aborted after %d steps and the tree having %d nodes" %
                  (steps, len(tree)))
            return None

    # Build the path.
    path = [goalnode]
    while path[0].parent is not None:
        path.insert(0, path[0].parent)

    # Report and return.
    print("Finished after %d steps and the tree having %d nodes" %
          (steps, len(tree)))
    tree_size = len(tree)
    return path


# Post process the path.
def PostProcess(path):
    i = 0
    while (i < len(path)-2):
        if path[i].connectsTo(path[i+2]):
            path.pop(i+1)
        else:
            i = i+1


######################################################################
#
#  Main Code
#
def main(seed_maze=False, visual=True):
    global tree_size
    global maze
    if seed_maze != False:
        maze = seed_maze
    
    # Report the parameters.
    print('Running with size ', maze.width, ' and ', maze.num_keys, ' keys.')

    # Create the figure.
    if visual:
        visual = Visualization()

    # Create the start/goal nodes.

    (xstart, ystart) = maze.start
    startnode = Node(xstart, ystart)

    (xgoal,  ygoal) = maze.goal
    goalnode  = Node(xgoal,  ygoal)

    # Generate and show keys
    keys = maze.get_keys()
    key_list = []
    for i in range(len(keys)):
        key_node = Node(keys[i][0] + 0.5, keys[i][1] + 0.5)
        key_list.append(key_node)
        if visual:
            visual.drawNode(key_node, color='green', marker='o')


    # Show the start/goal nodes.
    if visual:
        visual.drawNode(startnode, color='orange', marker='o')
        visual.drawNode(goalnode,  color='purple', marker='o')
        visual.show("Showing basic world")


    # Run the RRT planner.
    print("Running RRT...")
    path = rrt(startnode, goalnode, key_list, visual)

    # If unable to connect, just note before closing.
    if not path:
        if visual:
            visual.show("UNABLE TO FIND A PATH")
        return -1

    # Show the path.
    if visual:
        visual.drawPath(path, color='r', linewidth=2)
        visual.show("Showing the raw path")


    # Post process the path.
    PostProcess(path)

    # Show the post-processed path.
    if visual:
        visual.drawPath(path, color='b', linewidth=2)
        visual.show("Showing the post-processed path")

    return tree_size


if __name__== "__main__":
    main()