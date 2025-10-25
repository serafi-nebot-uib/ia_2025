from practica import agent, joc
import random


def main():
    # random.seed(1234567890)
    mida = (10, 10)
    mida = (5, 5)

    agents = [
        agent.Viatger(),
        agent.Viatger()
    ]

    lab = joc.Laberint(agents, mida_taulell=mida)
    lab.comencar()


if __name__ == "__main__":
    main()
