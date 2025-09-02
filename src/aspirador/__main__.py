from aspirador import joc_gui, agent


agents = [agent.AspiradorTaula()]

hab = joc_gui.Aspirador(agents)
hab.comencar()