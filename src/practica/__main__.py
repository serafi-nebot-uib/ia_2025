from practica import agent, joc


def main():
    mida = (10, 10)

    agents = [
        agent.Viatger(),
    ]

    lab = joc.Laberint(agents, mida_taulell=mida)
    lab.comencar()


if __name__ == "__main__":
    main()
