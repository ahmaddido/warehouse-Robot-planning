import random
import matplotlib.pyplot as plt

from map_sett import STATIONS, GRID_SIZE, ANIM_INTERVAL_MS, ANIMATE, SEED, SAFE_BUFFER, TASK_TIME, PROB, grid_build, BATTERY_MAX, CHARGE_RATE
from astar_tools import astar, travel_station_times
from vis_anim import build_path, animate_robot, job_start_marker

def sample_task_time(letter):
    lo, hi = TASK_TIME[letter]
    return random.uniform(lo, hi)

def max_task_time(letter):
    ri = TASK_TIME[letter][1]
    return ri


def job_20_task(T):
    """
    Simulate the jobs
    Input: T[(A,B)] = travel time between stations
    output: total time, final station, final battery
    """

    random.seed(SEED)

    curr = "S"

    battery = BATTERY_MAX
    total_time = 0.0
    station_seq = ["S"]

    def go_charge():
        """
        charging task
        """

        nonlocal curr, battery, total_time, station_seq

        if curr != "F":
            total_time += T[(curr, "F")]
            battery -= T[(curr, "F")]
            curr = "F"
            station_seq.append("F")

        needed = BATTERY_MAX - battery
        t_charge = needed / CHARGE_RATE
        total_time += t_charge
        battery = BATTERY_MAX

    def ensure_move(dest):
        """
        Battery safety rule: we only move if battery can cover curr to dest and then to F with safe buffer
        otherwise we go charge first
        """

        nonlocal battery

        need = T[(curr, dest)] + T[(dest, "F")] + SAFE_BUFFER
        if battery < need:
            go_charge()
    
    def ensure_task(task_letter):
        """
        battery safety rule for tasks:
        we only start task when battery covers
        max task time + time to F + safe buffer 
        otherwise we go charge first
        """

        nonlocal battery

        need = max_task_time(task_letter) + T[(curr, "F")] + SAFE_BUFFER
        if battery < need:
            go_charge()
    

    def goto(dest):
        nonlocal curr, battery, total_time, station_seq

        ensure_move(dest)

        total_time += T[(curr, dest)]
        battery -= T[(curr, dest)]
        curr = dest
        station_seq.append(dest)

    def repeat_until_success(letter, p_succ):
        """
        repeat task until success
        """

        nonlocal battery, total_time
        while True:
            ensure_task(letter)
            t = sample_task_time(letter)
            total_time += t
            battery -= t
            if random.random() < p_succ:
                break
    
    #main job loop
    for job in range(1, 21):
        if job % 2 == 1:
            ship_target = "D"
        else:
            ship_target = "E"

        goto("A")
        repeat_until_success("A", PROB["A"])
        goto("B")


        #Task B and if failed go to C
        ensure_task("B")
        tB = sample_task_time("B")
        total_time += tB
        battery -= tB
        okB = (random.random() < PROB["B"])

        if not okB:
            goto("G")
            ensure_task("G")
            tG = sample_task_time("G")
            total_time += tG
            battery -= tG
        

        #Task C
        goto("C")
        repeat_until_success("C", PROB["C"])
        goto(ship_target)

        #shipping loop
        while True:
            ensure_task(curr) #curr is either D or E
            tS = sample_task_time(curr)
            total_time += tS
            battery -= tS

            if random.random() < PROB["SHIP"]:
                break

            if curr != "E":
                goto("E")
    return total_time, curr, battery, station_seq


def main():
    grid = grid_build()

    #precimpute the travel times between stations
    T = travel_station_times(grid)

    # simulation of the 20 jobs
    total_time, final_station, final_battery, station_seq = job_20_task(T)

    print(f"Total time: {total_time:.2f} sec")
    #print(f"Final station: {final_station}")
    #print(f"Final battery: {final_battery:.2f}%")

    if ANIMATE:
        cell_path, leg_start = build_path(grid, station_seq)
        job_cell_markers = job_start_marker(station_seq, leg_start)
        anim = animate_robot(grid, cell_path, job_cell_markers, trail_len=5)
        plt.show()

if __name__ == "__main__":
    main()

