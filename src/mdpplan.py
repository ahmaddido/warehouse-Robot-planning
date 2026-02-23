import random
from dataclasses import dataclass
import map_sett
from astar_tools import astar, travelStation_times

#job phases
PICK = 0
INSP = 1
PACK = 2
SHIP = 3
JDONE = 4

@dataclass(frozen=True)
class State:
    location: str
    phase: int
    jType: int     #0=standard(D) / 1=express(E)
    rework: int    #0/1 (if INSP fails at B  ... must do G before C)
    batt: int      #battery bucket

BAT_STEP = 1.0  #1 second buckets


#battery helpers
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def batt_to_bucket(batt_sec):
    """
    convert battery sec to int
    """
    return int(clamp(int(batt_sec // BAT_STEP), 0, int(map_sett.BATTERY_MAX // BAT_STEP)))

def bucket_to_batt(bucket):
    """
    umgekehrt von batt_to bucket; convert from int bucket to battery sec"""
    return bucket * BAT_STEP




TASK_TIME = {
    "A": (1.0, 2.0),
    "B": (1.5, 2.5),
    "C": (1.5, 2.5),
    "D": (1.0, 2.0),
    "E": (1.0, 2.0),
    "G": (2.0, 4.0),
}
P_SUCC = {
    "A": 0.95,
    "B": 0.80,
    "C": 0.95,
    "D": 0.85,
    "E": 0.85,
    "G": 1.00,
    "SHIP": 0.85
}


TASK_MEAN = {k: 0.5 * (lo + hi) for k, (lo, hi) in TASK_TIME.items()}
TASK_MAX  = {k: hi for k, (_, hi) in TASK_TIME.items()}

def task_time_mean(st: str):
    return TASK_MEAN[st]

def task_time_max(st: str):
    return TASK_MAX[st]

def task_time_sample(st: str):
    lo, hi = TASK_TIME[st]
    return random.uniform(lo, hi)


#Travel time matrix
def build_travelTime_mat(grid):
    """
    returns T[(a, b)]"""
    return travelStation_times(grid)




#which station by phase
def req_Station(s):
    """
    returns which station based on given J phase
    """
    if s.phase == PICK:
        return "A"
    
    if s.phase == INSP:
        return "B"
    
    if s.phase == PACK:
        if s.rework == 1:
            return "G" 
        else:
            return "C"
        
    if s.phase == SHIP:
        if s.jType == 0:
            return "D" 
        else:
            return "E"
        
    return None



#Battery feasibility rules
def canReach_F(loc, batt_sec, T):
    #physically reach F before battery hits 0 (no buffer)
    return batt_sec >= T[(loc, "F")]

def moveTask_reachF(s, dest, T):
    """
    Safe check for moving to dest and do the task
    """
    b = bucket_to_batt(s.batt)
    move_t = T[(s.location, dest)]
    task_t = task_time_max(dest)
    toF_t  = T[(dest, "F")]
    return b >= (move_t + task_t + toF_t + map_sett.SAFE_BUFFER)

def can_do_task_here_then_reach_F(s, T):
    """
    Safe check for doing task now at current station
    """
    b = bucket_to_batt(s.batt)
    task_t = task_time_max(s.location)
    toF_t  = T[(s.location, "F")]
    return b >= (task_t + toF_t + map_sett.SAFE_BUFFER)



#recharging time to full 
def charge_time_to_full(batt_bucket):
    b_sec = bucket_to_batt(batt_bucket)
    needed = map_sett.BATTERY_MAX - b_sec
    if needed <= 0:
        return 0.0
    return needed / map_sett.CHARGE_RATE



#Transition model
#transition using expected mean task times
def step_mdp_expected(s, action, T):
    return step_mdp_core(s, action, T, task_time_func=task_time_mean)

#transition with uniform tasj times
def step_mdp_uniform(s, action, T):
    return step_mdp_core(s, action, T, task_time_func=task_time_sample)



def step_mdp_core(s, action, T, task_time_func):
    kind, arg = action

    if kind == "move":
        dest = arg
        travel_t = T[(s.location, dest)]
        new_b = batt_to_bucket(bucket_to_batt(s.batt) - travel_t)
        stateee = State(dest, s.phase, s.jType, s.rework, new_b)
        return [(1.0, stateee, travel_t)]

    if kind == "charge":
        if s.location != "F":
            return []
        t_charge = charge_time_to_full(s.batt)
        s2 = State("F", s.phase, s.jType, s.rework, batt_to_bucket(map_sett.BATTERY_MAX))
        return [(1.0, s2, t_charge)]

    if kind == "doTask":
        req = req_Station(s)
        if s.location != req:
            return []

        st = s.location
        t = task_time_func(st)
        p = P_SUCC[st]
        new_b = batt_to_bucket(bucket_to_batt(s.batt) - t)

        #to ensure A repeat / B rework / G to PACK / C repeat 
        if st == "A":
            return [(p, State("A", INSP, s.jType, 0, new_b), t),
                    (1.0 - p, State("A", PICK, s.jType, 0, new_b), t)]

        if st == "B":
            return [(p, State("B", PACK, s.jType, 0, new_b), t),
                    (1.0 - p, State("B", PACK, s.jType, 1, new_b), t)]

        if st == "G":
            return [(1.0, State("G", PACK, s.jType, 0, new_b), t)]

        if st == "C":
            return [(p, State("C", SHIP, s.jType, 0, new_b), t),
                    (1.0 - p, State("C", PACK, s.jType, 0, new_b), t)]

       
        #for shipping
        if st == "D":
            return [(p, State("D", JDONE, s.jType, 0, new_b), t),
                    (1.0 - p, State("D", SHIP, 1, 0, new_b), t)]

        if st == "E":
            return [(p, State("E", JDONE, s.jType, 0, new_b), t),
                    (1.0 - p, State("E", SHIP, 1, 0, new_b), t)]

    return []


def sample_transition(transitions):
    r = random.random()
    acc = 0.0
    for p, s2, c in transitions:
        acc += p
        if r <= acc:
            return s2, c
    return transitions[-1][1], transitions[-1][2]



#Feasible actions 
def feasible_actions(s, T):
    if s.phase == JDONE:
        return []

    b = bucket_to_batt(s.batt)
    acts = []

    #charging only in F
    if s.location == "F" and b < map_sett.BATTERY_MAX:
        acts.append(("charge", ""))

    req = req_Station(s)

    if s.location != req:
        if moveTask_reachF(s, req, T):
            acts.append(("move", req))
        if s.location != "F" and canReach_F(s.location, b, T):
            acts.append(("move", "F"))
        return acts

    if can_do_task_here_then_reach_F(s, T):
        acts.append(("doTask", ""))

    if s.location != "F" and canReach_F(s.location, b, T):
        acts.append(("move", "F"))

    return acts



# Value iteration
def optimalPolicy(T, theta=1e-4, max_iters=500):
    locations = list(map_sett.STATIONS.keys())
    max_bucket = int(map_sett.BATTERY_MAX // BAT_STEP)

    all_states = []
    for loc in locations:
        for phase in [PICK, INSP, PACK, SHIP, JDONE]:
            for jType in [0, 1]:
                for rework in [0, 1]:
                    for batt in range(max_bucket + 1):
                        all_states.append(State(loc, phase, jType, rework, batt))

    INF = 1e12
    V = {s: (0.0 if s.phase == JDONE else INF) for s in all_states}
    policy = {s: None for s in all_states}

    for _ in range(max_iters):
        delta = 0.0
        for s in all_states:
            if s.phase == JDONE:
                continue

            acts = feasible_actions(s, T)
            if not acts:
                continue

            best_a = None
            best_q = INF
            for a in acts:
                trans = step_mdp_expected(s, a, T)
                if not trans:
                    continue
                q = sum(p * (c + V[s2]) for p, s2, c in trans)
                if q < best_q:
                    best_q = q
                    best_a = a

            if best_a is None:
                continue

            delta = max(delta, abs(best_q - V[s]))
            V[s] = best_q
            policy[s] = best_a

        if delta < theta:
            break

    return policy


#greedy
def greedyPolicy(s, T):
    """
    pick the action with the smallest immediate expected costt
    """
    acts = feasible_actions(s, T)
    if not acts:
        return None
    best_a, best_q = None, float("inf")
    for a in acts:
        trans = step_mdp_expected(s, a, T)
        if not trans:
            continue
        q = sum(p * c for p, _, c in trans)  #immediate expected cost
        if q < best_q:
            best_q, best_a = q, a
    return best_a




#FASTevaluation simulation WITH TIMEOUTS
def simulate_20Jobs(
    T,
    policy=None,
    seed=1,
    max_steps_per_job=5000,        #800 wasnt enough for me 
    job_time_cap_sec=600.0,        #cap per job to avoid rare infinite loops
    episode_time_cap_sec=20000.0,  #cap per full 20-job episode
):
    """
    run 20 jobs and returm
      total_time, n_charges, n_reworks, n_ship_reroutes, timeout(bool)
    timeout=True if we exceed a cap (do not crash evaluation).
    """
    random.seed(seed)

    loc = "S"
    batt = batt_to_bucket(map_sett.BATTERY_MAX)

    total_time = 0.0
    n_charges = 0
    n_reworks = 0
    n_ship_reroutes = 0

    for job in range(1, 21):
        
        #every sec job is Express rest is standard
        if job % 2 == 1:
            jType = 0
        else:
            jType = 1

        s = State(loc, PICK, jType, 0, batt)

        steps = 0
        job_time = 0.0

        while s.phase != JDONE and steps < max_steps_per_job:
            if policy is None:
                a = greedyPolicy(s, T)
            else:
                a = policy.get(s, None) or greedyPolicy(s, T)

            #this should be rare due to the feasibility rules and checks
            if a is None:
                # emergency: try to go to F if possible
                bsec = bucket_to_batt(s.batt)
                if s.location != "F" and canReach_F(s.location, bsec, T):
                    a = ("move", "F")
                elif s.location == "F":
                    a = ("charge", "")
                else:
                    return total_time, n_charges, n_reworks, n_ship_reroutes, True

            trans = step_mdp_uniform(s, a, T)
            
            #if action fails for any reasin try to go charge
            if not trans:
                if s.location != "F":
                    a = ("move", "F")
                else:
                    a = ("charge", "")
                trans = step_mdp_uniform(s, a, T)
                if not trans:
                    return total_time, n_charges, n_reworks, n_ship_reroutes, True

            s2, dt = sample_transition(trans)

            total_time += dt
            job_time += dt
            steps += 1

            # caps (avoid rare long-tail non-termination in greedy)
            if job_time > job_time_cap_sec:
                return total_time, n_charges, n_reworks, n_ship_reroutes, True
            if total_time > episode_time_cap_sec:
                return total_time, n_charges, n_reworks, n_ship_reroutes, True

            kind, _ = a

            if kind == "charge":
                n_charges += 1

            if kind == "doTask" and s.location == "B" and s2.rework == 1:
                n_reworks += 1

            if kind == "doTask" and s.location == "D":
                if s2.location == "D" and s2.phase == SHIP and s2.jType == 1:
                    n_ship_reroutes += 1

            s = s2

        if s.phase != JDONE:
            return total_time, n_charges, n_reworks, n_ship_reroutes, True

        loc = s.location
        batt = s.batt

    return total_time, n_charges, n_reworks, n_ship_reroutes, False


#for the video
def simulate_20JobsAnim(
    T,
    grid,
    policy,
    seed=map_sett.SEED,
    max_steps_per_job=5000,
    job_time_cap_sec=600.0,
    episode_time_cap_sec=20000.0,
):

    random.seed(seed)

    #speed of the sim
    TASK_WAIT_FRAMES = 1
    CHARGE_WAIT_FRAMES = 1

    def wait_frames(cell_path, cell, n):
        for _ in range(n):
            cell_path.append(cell)

    #start state
    loc = "S"
    batt = batt_to_bucket(map_sett.BATTERY_MAX)

    cell_path = [map_sett.STATIONS[loc]]
    job_cell_markers = []
    frame_idx = 0

    total_time = 0.0

    for job in range(1, 21):

        if job % 2 == 1:
            jType = 0
        else:
            jType = 1

        s = State(loc, PICK, jType, 0, batt)

        job_cell_markers.append((frame_idx, job))

        steps = 0
        job_time = 0.0

        while s.phase != JDONE and steps < max_steps_per_job:
            # pick action from optimal policy; if missing, fallback to greedy
            a = policy.get(s, None) or greedyPolicy(s, T)

            # emergency fallback
            if a is None:
                bsec = bucket_to_batt(s.batt)
                if s.location != "F" and canReach_F(s.location, bsec, T):
                    a = ("move", "F")
                elif s.location == "F":
                    a = ("charge", "")
                #else:
                #    raise RuntimeError(f"Dead-end (no feasible actions) in animation sim at state: {s}")

            trans = step_mdp_uniform(s, a, T)

            #recovery if something invalid happened(rare)
            #if not trans:
            #    if s.location != "F":
            #        a = ("move", "F")
            #    else:
            #        a = ("charge", "")
            #    trans = step_mdp_uniform(s, a, T)
                #if not trans:
                #    raise RuntimeError(f"Recovery failed in animation sim at state: {s}")

            s2, dt = sample_transition(trans)

            total_time += dt
            job_time += dt
            steps += 1

            # caps (avoid infinite loops)
            #if job_time > job_time_cap_sec:
            #    raise RuntimeError(f"Job {job} exceeded job_time_cap_sec in animation sim. State: {s}")
            #if total_time > episode_time_cap_sec:
            #    raise RuntimeError(f"Episode exceeded episode_time_cap_sec in animation sim. State: {s}")

            kind, _ = a

            #for this actions frames 
            if kind == "move":
                u = map_sett.STATIONS[s.location]
                v = map_sett.STATIONS[s2.location]
                p = astar(grid, u, v)
                if p is None:
                    raise RuntimeError(f"No A* path between {s.location} and {s2.location}")
                # append cell-by-cell (skip first to avoid duplicates)
                for cell in p[1:]:
                    cell_path.append(cell)
                    frame_idx += 1

            elif kind == "doTask":
                #robot shpwn during task
                wait_frames(cell_path, map_sett.STATIONS[s.location], TASK_WAIT_FRAMES)
                frame_idx += TASK_WAIT_FRAMES

            elif kind == "charge":
                #robot shown during charging
                wait_frames(cell_path, map_sett.STATIONS["F"], CHARGE_WAIT_FRAMES)
                frame_idx += CHARGE_WAIT_FRAMES

            s = s2

        if s.phase != JDONE:
            raise RuntimeError(f"Job {job} not finisched in video. stopped at state: {s}")

        loc = s.location
        batt = s.batt

    return cell_path, job_cell_markers, total_time

