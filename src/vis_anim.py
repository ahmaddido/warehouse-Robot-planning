import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from map_sett import STATIONS, GRID_SIZE, ANIM_INTERVAL_MS
from astar_tools import astar


def build_path(grid, station_seq):
    """
    station seq is like ['S', 'A', 'B', 'C' ...]
    and we need to build a full path not a only station labels, so we run 
    astar between each pair of stations
    """

    full = []

    leg_start_cell_idx = [0]
    cell_i = 0

    for u, v in zip(station_seq[:-1], station_seq[1:]):
        p = astar(grid, STATIONS[u], STATIONS[v])
        if p is None:
            raise RuntimeError(f"No path found between {u} and {v}")
        
        # avoid duplicate points, since the end of one path is the start of the next
        if full:
            p = p[1:]
        full.extend(p)
        cell_i += len(p)
        leg_start_cell_idx.append(cell_i)
    
    return full, leg_start_cell_idx


def animate_robot(grid, cell_path, job_cell_markers, trail_len=10):
    """
    animate robot moving along cell path
    """

    H, W = GRID_SIZE
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_title("Warehouse Robot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # draw grid + shelves
    ax.imshow(grid, origin='lower', cmap='Blues', interpolation= "none", extent=[0, W, 0, H], zorder=0)

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.set_xticks(range(0, W + 1, 2))
    ax.set_yticks(range(0, H + 1, 2))
    ax.grid(True)

    # stations + robot + trails
    for letter, (x, y) in STATIONS.items():
        ax.scatter(x, y, c="red", s=120, zorder=5)
        ax.text(x, y, letter, color= "white", ha='center', va='center', zorder=6)
    
    robot, = ax.plot([], [], 'o', markersize=12, zorder=10)
    trail, = ax.plot([], [], '-', color='orange', linewidth=2, zorder=9)

    #job id
    job_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top', ha='left', zorder=20, fontsize=12, fontweight='bold')

    # convert path to x/y arrays
    xs = [ x for x, y in cell_path]
    ys = [ y for x, y in cell_path]

    marker_idx = 0
    current_job = 0

    def init():
        robot.set_data([], [])
        trail.set_data([], [])
        job_text.set_text("")
        return robot, trail, job_text
    
    def update(i):
        nonlocal marker_idx, current_job
        robot.set_data([xs[i]], [ys[i]])

        j0 = max(0, i - trail_len)
        trail.set_data(xs[j0:i+1], ys[j0:i+1])

        while marker_idx < len(job_cell_markers) and i >= job_cell_markers[marker_idx][0]:
            current_job = job_cell_markers[marker_idx][1]
            job_text.set_text(f"Job {current_job}")
            marker_idx += 1
        return robot, trail, job_text
    
    anim = FuncAnimation(
        fig, update, frames=len(cell_path), init_func=init, interval=ANIM_INTERVAL_MS, blit=True, repeat=False
    )
    return anim

def job_start_marker(station_seq, leg_start_cell_idx):
    """
    helper to mark the start of each job in the animation
    """

    markers = []
    job_num = 0

    for station_i, st in enumerate(station_seq):
        if st == "A":
            job_num += 1
            markers.append((leg_start_cell_idx[station_i], job_num))
    return markers