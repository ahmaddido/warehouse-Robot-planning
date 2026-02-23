import heapq
from typing import List, Tuple, Dict
from map_sett import SPEED_MPS, STATIONS

Cell = Tuple[int, int]
Path = List[Cell]
TravelTimes = Dict[Tuple[str, str], float]


def astar(grid, start, goal):
    """
    Astar returns list of (x,y) form start to goal // or None
    """
    H, W = grid.shape

    def heuristic(a, b):
        """
        admissible heuristic
        """
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        D = 1.0
        D2 = 2**0.5
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

    #the 8 connected moves
    neighbors = [
        (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, 2**0.5), (1, -1, 2**0.5), (-1, 1, 2**0.5), (-1, -1, 2**0.5)
    ]

    #the 4 connected moves
    # neighbors = [
    #     (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)
    # ]

    open_list = []
    heapq.heappush(open_list, (0.0, start))

    parent = {}
    g_cost = {start: 0.0}

    while open_list:
        _, cur = heapq.heappop(open_list)

        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]

        x, y = cur
        for dx, dy, step_cost in neighbors:
            nx = x + dx
            ny = y + dy

            #in bounds?
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            #free cell?
            if grid[ny, nx] == 1:
                continue
            #no corner cutting diagonal
            if dx != 0 and dy != 0:
                if grid[y, nx] == 1 or grid[ny, x] == 1:
                    continue

            nxt = (nx, ny)
            new_g = g_cost[cur] + step_cost

            if nxt not in g_cost or new_g < g_cost[nxt]:
                g_cost[nxt] = new_g
                parent[nxt] = cur
                f = new_g + heuristic(nxt, goal)
                heapq.heappush(open_list, (f, nxt))

    return None


def path_distance(path):
    """
    path length in grid m
    """
    if not path or len(path) < 2:
        return 0.0

    #1m for r/l/up/down ; wurzl 2 for diagonal 
    d = 0.0
    for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        d += (2**0.5) if (dx == 1 and dy == 1) else 1.0
    return d


def travelTime_sec(path):
    """
    form path length to time t=d/v
    """
    distance = path_distance(path)
    return distance / SPEED_MPS


def travelStation_times(grid):
    """
    precompute travel time between every 2 sation pair
    """
    names = list(STATIONS.keys())
    T: TravelTimes = {}

    for a in names:
        for b in names:
            if a == b:
                T[(a, b)] = 0.0
                continue

            p = astar(grid, STATIONS[a], STATIONS[b])
            if p is None:
                raise RuntimeError(f"Between {a} and {b} no path found ..... :/")

            T[(a, b)] = travelTime_sec(p)

    return T
