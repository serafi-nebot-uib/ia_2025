from practica import agent, joc
import random
import time

def run_dfs(mida: tuple[int, int]):
    agents = [agent.ViatgerDFS()]
    lab = joc.Laberint(agents, mida_taulell=mida)
    start = time.perf_counter()
    lab._logica(lab._agents)
    end = time.perf_counter()
    return end - start, agents[0].npases, agents[0].nestats

def run_astar(mida: tuple[int, int]):
    agents = [agent.ViatgerAstar()]
    lab = joc.Laberint(agents, mida_taulell=mida)
    start = time.perf_counter()
    lab._logica(lab._agents)
    end = time.perf_counter()
    return end - start, agents[0].npases, agents[0].nestats

def run_minimax(mida: tuple[int, int], poda: bool):
    agents = [agent.ViatgerMinimax(poda=poda), agent.ViatgerMinimax(poda=poda)]
    lab = joc.Laberint(agents, mida_taulell=mida)
    start = time.perf_counter()
    lab._logica(lab._agents)
    lab._logica(lab._agents)
    end = time.perf_counter()
    return end - start, tuple(a.npases for a in agents), tuple(a.nestats for a in agents)

def main():
    dfs_time, dfs_step, dfs_state = [], [], []
    as_time, as_step, as_state = [], [], []
    mm_time, mm_step, mm_state = [], [], []
    for i in range(30):
        try:
            print(f"iteration #{i}")
            state = random.getstate()

            random.setstate(state)
            tdfs, ndfs, edfs = run_dfs(mida=(10, 10))
            print(f"    DFS: {tdfs:10.6f} (pases: {ndfs}; estats: {edfs})")

            random.setstate(state)
            tas, nas, eas = run_astar(mida=(10, 10))
            print(f"  astar: {tas:10.6f} (pases: {nas}; estats: {eas})")

            random.setstate(state)
            tmm, nmm, emm = run_minimax(mida=(5, 5), poda=True)
            print(f"minimax: {tmm:10.6f} (pases: {', '.join(map(str, nmm))}; estats: {', '.join(map(str, emm))})")

            dfs_time.append(tdfs)
            dfs_step.append(ndfs)
            dfs_state.append(edfs)
            as_time.append(tas)
            as_step.append(nas)
            as_state.append(eas)
            mm_time.append(tmm)
            mm_step.append(nmm)
            mm_state.append(emm)
        except KeyboardInterrupt:
          pass

    print("########## RESULTS ##########")
    print("DFS")
    print(",".join(map(str, dfs_time)))
    print(",".join(map(str, dfs_step)))
    print(",".join(map(str, dfs_state)))
    print("ASTAR")
    print(",".join(map(str, as_time)))
    print(",".join(map(str, as_step)))
    print(",".join(map(str, as_state)))
    print("MINIMAX")
    print(",".join(map(str, mm_time)))
    print(",".join(map(str, (a[0] for a in mm_step))))
    print(",".join(map(str, (a[1] for a in mm_step))))
    print(",".join(map(str, (a[0] for a in mm_state))))
    print(",".join(map(str, (a[1] for a in mm_state))))

if __name__ == "__main__":
    main()
