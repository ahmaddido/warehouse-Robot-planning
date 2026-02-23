import matplotlib.pyplot as plt
import map_sett
from map_sett import grid_build
from vis_anim import video_robot
from mdpplan import build_travelTime_mat, optimalPolicy, simulate_20JobsAnim


def main():
    grid = grid_build()
    T = build_travelTime_mat(grid)

    print("starting the optimal policy.....:D")
    policy = optimalPolicy(T)
    print("Policy computed.")

    cell_path, job_markers, total_time = simulate_20JobsAnim(T, grid, policy=policy, seed=map_sett.SEED)
    print(f"Total makespan: {total_time:.2f} sec")

    if map_sett.ANIMATE:
        anim = video_robot(grid, cell_path, job_markers, trail_len=5)
        plt.show()


if __name__ == "__main__":
    main()
