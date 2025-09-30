from quiques import agent_amplada, agent_profunditat, joc


def main():
    # barca = agent_amplada.BarcaAmplada()
    barca = agent_profunditat.BarcaProfunditat()
    illes = joc.Joc([barca])
    illes.comencar()


if __name__ == "__main__":
    main()
