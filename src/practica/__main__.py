from practica import agent, joc

def run_dfs(mida: tuple[int, int]):
    print("DFS")
    agents = [agent.ViatgerDFS()]
    lab = joc.Laberint(agents, mida_taulell=mida)
    lab.comencar()

def run_astar(mida: tuple[int, int]):
    print("A*")
    agents = [agent.ViatgerAstar()]
    lab = joc.Laberint(agents, mida_taulell=mida)
    lab.comencar()

def run_minimax(mida: tuple[int, int], poda: bool, cache: bool):
    print("MINIMAX")
    agents = [agent.ViatgerMinimax(poda=poda, cache=cache), agent.ViatgerMinimax(poda=poda, cache=cache)]
    lab = joc.Laberint(agents, mida_taulell=mida)
    lab.comencar()

def main():
    # run_dfs(mida=(10, 10))
    # run_astar(mida=(10, 10))
    run_minimax(mida=(5, 5), poda=True, cache=False)

if __name__ == "__main__":
    main()