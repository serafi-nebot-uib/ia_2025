""" Fitxer que conté l'agent barca en profunditat.

S'ha d'implementar el mètode:
    actua()
"""
from quiques.agent import Barca
from quiques.estat import Estat


class BarcaProfunditat(Barca):
    def __init__(self):
        super(BarcaProfunditat, self).__init__()

    def actua(self, percepcio: dict) -> str | tuple[str, (int, int)]:
        pass
