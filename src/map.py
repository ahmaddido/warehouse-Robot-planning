import numpy as np
import matplotlib.pyplot as plt
import heapq

grid_size = (16, 18)
grid = np.zeros(grid_size, dtype=int)

#shelves and stations
grid[4:12, 2:4] = 1
grid[4:12, 6:8] = 1
grid[4:12, 10:12] = 1
grid[4:12, 14:16] = 1

stations = {
    "S": (1, 13),
    "A": (5, 15),
    "B": (9, 15),
    "C": (13, 15),
    "D": (17, 15),
    "E": (13, 1),
    "F": (9, 1),
    "G": (5, 1),
}

plt.figure(figsize=(6, 6))
plt.title('warehouse problem')
plt.xlabel('x')
plt.ylabel('y')

for letter,(x, y) in stations.items():
    plt.scatter(x, y, c='red', s=100)
    plt.text(x, y, letter, color='white', ha='center', va='center')

plt.grid(True)




def astar(grid, start, goal):
    width = grid.shape[1]
    height = grid.shape[0]

    def heuristic(a, b):
        dx = abs(a[0]-b[0])
        dy = abs(a[1]-b[1])
        D = 1
        D2 = 2**0.5
        return D*(dx+dy) + (D2-2*D)*min(dx,dy)

    neighbors = [
        (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
        (1, 1, 2**0.5), (1, -1, 2**0.5), (-1, 1, 2**0.5), (-1, -1, 2**0.5)
    ]

    open_list = []
    heapq.heappush(open_list, (0, start))

    parent = {}
    cost_g = {start: 0}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = [current]
            while current in parent:
                current = parent[current]
                path.append(current)
            return path[::-1]

        x, y = current
        for dx, dy, cost in neighbors:
            nx, ny = x + dx, y + dy

            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if grid[ny][nx] == 1:
                continue

            # prevent corner cutting on diagonals
            if dx != 0 and dy != 0:
                if grid[y][nx] == 1 or grid[ny][x] == 1:
                    continue

            neighbor = (nx, ny)
            g_estimate = cost_g[current] + cost

            if neighbor not in cost_g or g_estimate < cost_g[neighbor]:
                parent[neighbor] = current
                cost_g[neighbor] = g_estimate
                f_score = g_estimate + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score, neighbor))

    return None




plt.xticks(range(0, grid_size[1], 2))
plt.yticks(range(0, grid_size[0], 2))

#plt.xticks(range(0, grid_size[1]+1, 1))
#plt.yticks(range(0, grid_size[0]+1, 1))

plt.imshow(grid, origin='lower', cmap='Blues', extent=[0, grid_size[1], 0, grid_size[0]])

plt.xlim(0, grid_size[1])
plt.ylim(0, grid_size[0])


path = astar(grid, stations["S"], stations["E"])
print("Path length:", len(path) if path else None)

if path:
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    plt.plot(xs, ys, linewidth=3)

plt.show()