""" Fitxer que conté l'agent barca en amplada implementat.

Author: Miquel Miró Nicolau (UIB), 2024
"""

from quiques.agent import Barca
from quiques.estat import Estat


class BarcaAmplada(Barca):
    def __init__(self):
        super(BarcaAmplada, self).__init__()
        self.__frontera = None
        self.__tancats = None
        self.__cami_exit = None
        self.nom = ""

    def cerca(self, estat_inicial: Estat) -> bool:
        self.__frontera = []
        self.__tancats = set()
        exit = False

        self.__frontera.append(estat_inicial)
        while self.__frontera:
            estat_actual = self.__frontera.pop(0)

            if estat_actual in self.__tancats or not estat_actual.es_segur():
                continue

            if estat_actual.es_meta():
                break

            for f in estat_actual.genera_fill():
                self.__frontera.append(f)

            self.__tancats.add(estat_actual)

        if estat_actual.es_meta():
            self.__cami_exit = estat_actual.cami
            exit = True

        return exit

    def actua(self, percepcio: dict) -> tuple[str, (int, int)]:
        if self.__cami_exit is None:
            estat_inicial = Estat(
                local_barca=percepcio["Lloc"],
                llops_esq=percepcio["Llop Esq"],
                polls_esq=percepcio["Poll Esq"],
            )

            self.cerca(estat_inicial)

        if self.__cami_exit:
            quiques, llops = self.__cami_exit.pop(0)

            return "M", (quiques, llops)
        else:
            return "A", None
