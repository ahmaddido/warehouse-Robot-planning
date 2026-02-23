import math
import statistics
from map_sett import grid_build
from mdpplan import build_travelTime_mat, optimalPolicy, simulate_20Jobs

def mean_ci95(xs):
    n = len(xs)
    mu = statistics.mean(xs)
    sd = statistics.pstdev(xs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    ci = 1.96 * se
    return mu, sd, (mu - ci, mu + ci)

def run_policy(name, T, policy, n_runs, seed0):
    """
    to simulate many runs and print summary stats
    """
    makespans = []
    charges = []
    reworks = []
    reroutes = []
    timeouts = 0

    for i in range(n_runs):
        seed = seed0 + i
        total_time, n_ch, n_rw, n_rr, timeout = simulate_20Jobs(
            T,
            policy=policy,
            seed=seed,
            max_steps_per_job=5000,
            job_time_cap_sec=600.0,
            episode_time_cap_sec=20000.0,
        )
        if timeout:
            timeouts += 1
            continue

        makespans.append(total_time)
        charges.append(n_ch)
        reworks.append(n_rw)
        reroutes.append(n_rr)

    if not makespans:
        print(f"{name}: all runs timed out")
        return

    mu, sd, (lo, hi) = mean_ci95(makespans)

    print(f"\n{name}")
    print(f"  runs: {len(makespans)} (timeouts: {timeouts})")
    print(f"  makespan mean: {mu:.2f} s ")
    print(f"  charges mean:  {statistics.mean(charges):.2f}")
    print(f"  reworks mean:  {statistics.mean(reworks):.2f}")
    print(f"  reroutes mean: {statistics.mean(reroutes):.2f}")




def main():
    grid = grid_build()
    T = build_travelTime_mat(grid)

    opt_policy = optimalPolicy(T)

    run_policy("Optimal", T, opt_policy, n_runs=500, seed0=1)
    run_policy("Greedy", T, policy=None, n_runs=500, seed0=1)

    

if __name__ == "__main__":
    main()
