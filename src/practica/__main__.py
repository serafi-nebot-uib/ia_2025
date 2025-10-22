from practica import agent, joc


def main():
    minimax = agent.MODE == 2

    mida = (5, 5) if minimax else (10, 10)
    agents = [agent.Viatger() for _ in range(2 if minimax else 1)]

    lab = joc.Laberint(agents, mida_taulell=mida)
    lab.comencar()


if __name__ == "__main__":
    main()
