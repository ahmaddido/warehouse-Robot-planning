import numpy as np

SEED = 1
ANIMATE = True
ANIM_INTERVAL_MS = 200

# each grid step as 1m
SPEED_MPS = 2.0
BATTERY_MAX = 100.0
CHARGE_RATE = 10.0

#battery safe buffer to avoid cutting 
SAFE_BUFFER = 3.0

GRID_SIZE = (16, 18)

STATIONS = {
    'S': (1, 13),
    'A': (5, 15),
    'B': (9, 15),
    'C': (13, 15),
    'D': (17, 15),
    'E': (13, 1),
    'F': (9, 1),
    'G': (5, 1),
}

#TASK_TIME = {
#    'A': (1.0, 2.0),
#    'B': (1.0, 3.0),
#    'C': (2.0, 4.0),
#    'D': (1.0, 3.0),
#    'E': (1.0, 3.0),
#    'G': (2.0, 3.0),
#}

#PROB = {"A": 0.95, 
#        "B": 0.80, 
#        "C": 0.90, 
#        "D": 0.85, 
#        "E": 0.85, 
#        "F": 1.0, 
#        "G": 1.0, 
#        "SHIP": 0.85
#        }


def grid_build():
    """
    to create the warehouse grid with shelves
    """
    H, W = GRID_SIZE
    grid = np.zeros((H, W), dtype=int)

    #shelves
    grid[4:12, 2:4] = 1
    grid[4:12, 6:8] = 1
    grid[4:12, 10:12] = 1
    grid[4:12, 14:16] = 1

    return grid



