from practica import agent, joc

import random

def main():
    # random.seed(1337)

    mida = (10, 10)

    agents = [
        agent.Viatger(),
    ]

    lab = joc.Laberint(agents, mida_taulell=mida)
    lab.comencar()


if __name__ == "__main__":
    main()
