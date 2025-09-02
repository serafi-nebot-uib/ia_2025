from iaLib import agent, joc

class AspiradorNoG(joc.JocNoGrafic):

    def __init__(self, agents: list[agent.Agent] | None = None):
        if agents is None:
            agents = []
        super(AspiradorNoG, self).__init__(agents=agents)
        # TODO


    def _draw(self):
        # TODO
        pass


    def percepcio(self):
        # TODO
        pass


    def _aplica(self, accio, params=None, agent_actual=None):
        # TODO
        pass

