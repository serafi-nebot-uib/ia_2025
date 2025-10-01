""" Mòdul que conté l'agent per jugar al joc de les monedes.

Percepcions:
    ClauPercepcio.MONEDES
Solució:
    " XXXC"
"""

from typing import Self
from iaLib import agent
from copy import deepcopy

SOLUCIO = " XXXC"

class Estat:
    def __init__(self, monedes: list[str] | str, cami: list[tuple[str, int]] | None = None):
        assert len(monedes) == len(SOLUCIO)
        self.monedes: list[str] = list(monedes)
        self.cami: list[tuple[str, int]] | None = cami

        self.__p = monedes.index(" ")
        assert self.__p >= 0, f"combinació de monedes invalids: {monedes}"

        self.__p0 = abs(SOLUCIO.find(" ") - self.__p)
        self.__v = sum(int(e != s) for s, e in zip(SOLUCIO, monedes) if e != " ")
        self.__h = self.__p0 + self.__v

    @property
    def p(self) -> int: return self.__p
    @property
    def h(self) -> int: return self.__h

    def desp(self, id_moneda: int) -> Self | None:
        m, c = deepcopy(self.monedes), self.cami if self.cami is not None else []
        if 0 <= id_moneda < len(self.monedes) and id_moneda in (self.__p - 1, self.__p + 1):
            m[self.__p], m[id_moneda] = m[id_moneda], m[self.__p]
            return self.__class__(m, c + [("D", id_moneda)])
        return None

    def gira(self, id_moneda: int) -> Self | None:
        m, c = deepcopy(self.monedes), self.cami if self.cami is not None else []
        if 0 <= id_moneda < len(self.monedes) and id_moneda != self.__p:
            m[id_moneda] = "C" if m[id_moneda] == "X" else "X"
            return self.__class__(m, c + [("G", id_moneda)])
        return None

    def bota(self, id_moneda: int) -> Self | None:
        m, c = deepcopy(self.monedes), self.cami if self.cami is not None else []
        if 0 <= id_moneda < len(self.monedes) and id_moneda in (self.__p - 2, self.__p + 2):
            m[id_moneda] = "C" if m[id_moneda] == "X" else "X"
            m[self.__p], m[id_moneda] = m[id_moneda], m[self.__p]
            return self.__class__(m, c + [("B", id_moneda)])
        return None

    accions = { "D": desp, "G": gira, "B": bota }

    def transicio(self, accio: str, id_moneda: int) -> Self | None: return self.accions.get(accio, lambda *_: None)(self, id_moneda)

    def fills(self) -> list[Self]:
        accions = [("D", i) for i in (self.p - 1, self.p + 1)] + [("G", i) for i in range(len(self.monedes))] + [("B", i) for i in (self.p - 2, self.p + 2)]
        return list(filter(None, (self.transicio(*args) for args in accions)))

    def __str__(self) -> str: return f"{''.join(self.monedes)} ({self.__p0} + {self.__v})"
    def __repr__(self) -> str: return str(self)
    def __hash__(self) -> int: return hash(tuple(self.monedes))
    def __eq__(self, other) -> bool: return self.monedes == other.monedes

class AgentMoneda(agent.Agent):
    def __init__(self):
        super().__init__(long_memoria=0)
        self.__oberts: list[Estat] = []
        self.__tancats: set[Estat] = set()
        self.__accions: list[tuple[str, int]] | None = None
        self.nom = ""

    def pinta(self, display):
        print(self._posicio_pintar)

    def cerca(self, estat_inicial: Estat) -> bool:
        self.__oberts = []
        self.__tancats = set()
        estat_actual: Estat | None = None
        exit = False

        self.__oberts.append(estat_inicial)
        while self.__oberts:
            estat_actual = self.__oberts.pop(-1)

            if estat_actual in self.__tancats: continue
            if estat_actual.h == 0: break

            self.__oberts += estat_actual.fills()
            self.__tancats.add(estat_actual)

        if estat_actual and estat_actual.h == 0:
            self.__accions = estat_actual.cami
            exit = True

        return exit

    def actua(self, percepcio) -> tuple[str, int | None]:
        if self.__accions is None:
            self.cerca(Estat(percepcio["Monedes"]))
            print(f"n steps: {len(self.__accions if self.__accions else [])}")
        return self.__accions.pop(0) if self.__accions else ("R", None)