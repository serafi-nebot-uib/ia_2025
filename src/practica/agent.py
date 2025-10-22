from practica import joc
from practica.estat import Pos, Estat
from queue import PriorityQueue

class Viatger(joc.Viatger):
    def __init__(self, *args, **kwargs):
        super(Viatger, self).__init__(*args, **kwargs)
        self.__accions = None

    def dfs(self, estat_inicial: Estat) -> bool:
        oberts = []
        tancats = set()
        estat_actual: Estat | None = None
        exit = False

        oberts.append(estat_inicial)
        while oberts:
            estat_actual = oberts.pop(-1)

            if estat_actual is None: break
            if estat_actual in tancats: continue
            if estat_actual.h == 0: break

            for f in estat_actual.fills(): oberts.append(f)
            tancats.add(estat_actual)

        if estat_actual and estat_actual.h == 0:
            self.__accions = estat_actual.cami
            exit = True

        return exit

    def astar(self, estat_inicial: Estat) -> bool:
        oberts = PriorityQueue()
        tancats = set()
        estat_actual: Estat | None = None
        exit = False

        oberts.put(estat_inicial)
        while oberts:
            estat_actual = oberts.get()

            if estat_actual is None: break
            if estat_actual in tancats: continue
            if estat_actual.h == 0: break

            for f in estat_actual.fills(): oberts.put(f)
            tancats.add(estat_actual)

        if estat_actual and estat_actual.h == 0:
            self.__accions = estat_actual.cami
            exit = True

        return exit

    def actua(self, percepcio) -> tuple[str, str]:
        if self.__accions is None:
            dim = percepcio["MIDA"]
            pos = Pos(*percepcio["AGENTS"][percepcio["TORN"]])
            parets = {Pos(*p) for p in percepcio["PARETS"]}
            desti = Pos(*percepcio["DESTI"])
            # self.dfs(Estat(pos, desti, parets, dim))
            self.astar(Estat(pos, desti, parets, dim))
            print(f"n steps: {len(self.__accions if self.__accions else [])}")
        if self.__accions:
            accio = self.__accions.pop(0)
            print(accio)
            return accio
        return "ESPERAR", ""