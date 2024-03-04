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
from generators.maze_generator import generate_maze

######################################################################
#
#   Parameters
#
#   Define the step size.  Also set the maximum number of nodes.
#
DSTEP = 1

# Maximum number of steps (attempts) or nodes (successful steps).
SMAX = 50000
NMAX = 1500


######################################################################
#
#   World Definitions
#
#   List of obstacles/objects as well as the start/goal.
#
difficulty = 1.
(xmin, xmax) = (0, 41)
(ymin, ymax) = (0, 41)
maze = generate_maze(xmax, ymax)

# Collect all the triangle and prepare (for faster checking).

polys = []
for i in range(xmax):
    for j in range(ymax):
        if maze[j, i] > 0 and random.random() < difficulty:
            if (i - 1 >= 0) and maze[j, i - 1] > 0:
                polys.append(Polygon([[i, j+0.45], [i+0.5, j+0.4], [i+0.5, j+0.55], [i, j+0.55]]))
            if (i + 1 < xmax) and maze[j, i + 1] > 0:
                polys.append(Polygon([[i + 0.5, j+0.45], [i+1, j+0.45], [i+1, j+0.55], [i+0.5, j+0.55]]))
            if (j - 1 >= 0) and maze[j - 1, i] > 0:
                polys.append(Polygon([[i+0.45, j], [i+0.45, j+0.5], [i+0.55,j+0.5], [i+0.55,j]]))
            if (j + 1 < ymax) and maze[j + 1, i] > 0:
                polys.append(Polygon([[i+0.45, j+0.5], [i+0.45, j+1], [i+0.55,j+1], [i+0.55,j+0.5]]))
filled_grids = prep(MultiPolygon(polys))

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

        # Show the triangles.
        for poly in filled_grids.context.geoms:
            plt.plot(*poly.exterior.xy, 'k-', linewidth=2)

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
        self.directions = ['left', 'right', 'up', 'down']

    ############
    # Utilities:
    # In case we want to print the node.
    def __repr__(self):
        return ("<Point %5.2f,%5.2f>" % (self.x, self.y))

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

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
        return filled_grids.disjoint(Point(self.coordinates()))

    # Check the local planner - whether this connects to another node.
    def connectsTo(self, other):
        line = LineString([self.coordinates(), other.coordinates()])
        return filled_grids.disjoint(line)


######################################################################
#
#   RRT Functions
#
def rrt(startnode, goalnode, visual):
    # Start the tree with the startnode (set no parent just in case).
    startnode.parent = None
    tree = [startnode]

    # Function to attach a new node to an existing node: attach the
    # parent, add to the tree, and show in the figure.
    def addtotree(oldnode, newnode):
        newnode.parent = oldnode
        tree.append(newnode)
        visual.drawEdge(oldnode, newnode, color='g', linewidth=1)
        visual.show()

    # Loop - keep growing the tree.
    steps = 0
    found_goal = False
    while not found_goal:
        # Directly determine the distances to the goal node.
        distances = np.array([node.distance(goalnode) for node in tree])
        index     = np.argmin(distances)
        nearnode  = tree[index]
        d         = distances[index]

        if len(nearnode.directions) != 0:
            direction = random.choice(nearnode.directions)
            if direction == 'left':
                nextnode = Node(nearnode.x - DSTEP, nearnode.y)
                nextnode.directions.remove('right')
            elif direction == 'right':
                nextnode = Node(nearnode.x + DSTEP, nearnode.y)
                nextnode.directions.remove('left')
            elif direction == 'up':
                nextnode = Node(nearnode.x, nearnode.y + DSTEP)
                nextnode.directions.remove('down')
            elif direction == 'down':
                nextnode = Node(nearnode.x, nearnode.y - DSTEP)
                nextnode.directions.remove('up')

            nearnode.directions.remove(direction)

            # Check whether to attach.
            if nextnode.inFreespace() and nearnode.connectsTo(nextnode) and nextnode not in tree:
                addtotree(nearnode, nextnode)

                # If within DSTEP, also try connecting to the goal.  If
                # the connection is made, break the loop to stop growing.
                if nextnode.distance(goalnode) < DSTEP and nextnode.connectsTo(goalnode):
                    addtotree(nextnode, goalnode)
                    found_goal = True
        else:
            del tree[index]

        # Check whether we should abort - too many steps or nodes.
        steps += 1
        if (steps >= SMAX) or (len(tree) >= NMAX):
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
def main():
    # Report the parameters.
    print('Running with step size ', DSTEP, ' and up to ', NMAX, ' nodes.')

    # Create the figure.
    visual = Visualization()

    # Create the start/goal nodes.

    (xstart, ystart) = (random.uniform(xmin, xmax), random.uniform(ymin, ymax))
    startnode = Node(xstart, ystart)
    while not startnode.inFreespace():
        (xstart, ystart) = (random.uniform(xmin, xmax), random.uniform(ymin, ymax))
        startnode = Node(xstart, ystart)

    (xgoal,  ygoal)  = (random.uniform(xmin, xmax), random.uniform(ymin, ymax))
    goalnode  = Node(xgoal,  ygoal)
    while not goalnode.inFreespace():
        (xgoal,  ygoal)  = (random.uniform(xmin, xmax), random.uniform(ymin, ymax))
        goalnode  = Node(xgoal,  ygoal)

    # Show the start/goal nodes.
    visual.drawNode(startnode, color='orange', marker='o')
    visual.drawNode(goalnode,  color='purple', marker='o')
    visual.show("Showing basic world")


    # Run the RRT planner.
    print("Running RRT...")
    path = rrt(startnode, goalnode, visual)

    # If unable to connect, just note before closing.
    if not path:
        visual.show("UNABLE TO FIND A PATH")
        return

    # Show the path.
    visual.drawPath(path, color='r', linewidth=2)
    visual.show("Showing the raw path")


    # Post process the path.
    PostProcess(path)

    # Show the post-processed path.
    visual.drawPath(path, color='b', linewidth=2)
    visual.show("Showing the post-processed path")


if __name__== "__main__":
    main()
