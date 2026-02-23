import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from map_sett import GRID_SIZE, ANIM_INTERVAL_MS, STATIONS


def video_robot(grid, cell_path, job_cell_markers, trail_len=6):
    H, W = GRID_SIZE
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_title("Warehouse Robot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.imshow(grid, origin='lower', cmap='Blues', interpolation="none", extent=[0, W, 0, H], zorder=0)
  

    #ax.set_xlim(0, W)
    #ax.set_ylim(0, H)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    #ax.set_xticks(range(0, W + 1, 2))
    #ax.set_yticks(range(0, H + 1, 2))
    

    ax.set_xticks(range(0, W + 1, 2))
    ax.set_yticks(range(0, H + 1, 2))
    ax.grid(True)

    #ax.set_xticks([i - 0.5 for i in range(0, W + 1)])
    #ax.set_yticks([i - 0.5 for i in range(0, H + 1)])
    #ax.grid(True)

    

    #for stations letters
    for letter, (x, y) in STATIONS.items():
    #     #ax.scatter(x + 0.5, y + 0.5, c="red", s=120, zorder=5)
    #     #ax.text(x + 0.5, y + 0.5, letter, color="white", ha='center', va='center', zorder=6)
        
        ax.scatter(x, y, c="red", s=120, zorder=5)
        ax.text(x, y, letter, color="white", ha='center', va='center', zorder=6)

    # stations
    # for letter, (x, y) in STATIONS.items():
    #     ax.scatter(x + 1, y + 1, c="red", s=120, zorder=5)
    #     ax.text(x + 1, y + 1, letter, color="white", ha='center', va='center', zorder=6)


# # robot path
#     xs = [x + 1 for x, y in cell_path]
#     ys = [y + 1 for x, y in cell_path]



    #robot and trails
    robot, = ax.plot([], [], 'o', markersize=12, zorder=10)
    trail, = ax.plot([], [], '-', color='orange', linewidth=2, zorder=9)

    job_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va='top', ha='left', zorder=20, fontsize=12, fontweight='bold')

    #xs = [x + 0.5 for x, y in cell_path]
    #ys = [y + 0.5 for x, y in cell_path]

    xs = [x for x, y in cell_path]
    ys = [y for x, y in cell_path]

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

        #job nr update on screen
        while marker_idx < len(job_cell_markers) and i >= job_cell_markers[marker_idx][0]:
            current_job = job_cell_markers[marker_idx][1]
            job_text.set_text(f"Job {current_job}")
            marker_idx += 1

        return robot, trail, job_text

    anim = FuncAnimation(
        fig, update,
        frames=len(cell_path),
        init_func=init,
        interval=ANIM_INTERVAL_MS,
        blit=False,
        repeat=False
    )
    return anim
