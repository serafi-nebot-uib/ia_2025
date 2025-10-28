from practica import joc
from practica.estat import Pos, Estat, EstatAdv
from queue import PriorityQueue
import time

class ViatgerDFS(joc.Viatger):
    def __init__(self, *args, **kwargs):
        super(ViatgerDFS, self).__init__(*args, **kwargs)
        self.__accions = None
        self.npases = 0

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
            tancats.add(estat_actual)
            if estat_actual.h == 0: break

            for f in estat_actual.fills:
                if f not in tancats:
                    oberts.append(f)

        if estat_actual and estat_actual.h == 0:
            self.__accions = estat_actual.cami
            exit = True

        return exit

    def actua(self, percepcio) -> tuple[str, str]:
        dim = percepcio["MIDA"]
        torn = percepcio["TORN"]
        pos = Pos(*percepcio["AGENTS"][torn])
        parets = {Pos(*p) for p in percepcio["PARETS"]}
        desti = Pos(*percepcio["DESTI"])

        if self.__accions is None:
            start = time.perf_counter()
            e = Estat(pos, desti, parets, dim)
            self.dfs(e)
            end = time.perf_counter()
            self.npases = len(self.__accions if self.__accions else [])
            # print(f"{self.nom} ha trobat solució en {end - start:.6f} segons ({self.npases} pases)")

        if self.__accions: return self.__accions.pop(0)
        return "ESPERAR", ""

class ViatgerAstar(joc.Viatger):
    def __init__(self, *args, **kwargs):
        super(ViatgerAstar, self).__init__(*args, **kwargs)
        self.__accions = None
        self.npases = 0

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
            tancats.add(estat_actual)
            if estat_actual.h == 0: break

            for f in estat_actual.fills:
                if f not in tancats:
                    oberts.put(f)

        if estat_actual and estat_actual.h == 0:
            self.__accions = estat_actual.cami
            exit = True

        return exit

    def actua(self, percepcio) -> tuple[str, str]:
        dim = percepcio["MIDA"]
        torn = percepcio["TORN"]
        pos = Pos(*percepcio["AGENTS"][torn])
        parets = {Pos(*p) for p in percepcio["PARETS"]}
        desti = Pos(*percepcio["DESTI"])

        if self.__accions is None:
            start = time.perf_counter()
            e = Estat(pos, desti, parets, dim)
            self.astar(e)
            end = time.perf_counter()
            self.npases = len(self.__accions if self.__accions else [])
            # print(f"{self.nom} ha trobat solució en {end - start:.6f} segons ({self.npases} pases)")

        if self.__accions: return self.__accions.pop(0)
        return "ESPERAR", ""


class ViatgerMinimax(joc.Viatger):
    def __init__(self, *args, **kwargs):
        super(ViatgerMinimax, self).__init__(*args, **kwargs)
        self.__accions = None
        self.npases = 0

    def minimax(self, estat: EstatAdv, alpha: float | None = None, beta: float | None = None) -> EstatAdv:
        if estat.meta: return estat

        fills = []
        prune = None not in (alpha, beta)
        for fill in estat.fills:
            final = self.minimax(fill, alpha, beta)
            if prune:
                if estat.torn: alpha = max(alpha, final.puntuacio)
                else: beta = min(beta, final.puntuacio)
            fills.append(final)
            if prune and alpha >= beta: break

        fn = max if estat.torn else min
        return fn(fills, key=lambda x: x.puntuacio)

    def actua(self, percepcio) -> tuple[str, str]:
        dim = percepcio["MIDA"]
        torn = percepcio["TORN"]
        pos = Pos(*percepcio["AGENTS"][torn])
        adv = next((Pos(*v) for k, v in percepcio["AGENTS"].items() if k != torn), None)
        assert adv is not None, "no hi ha posició de l'adversari"
        parets = {Pos(*p) for p in percepcio["PARETS"]}
        desti = Pos(*percepcio["DESTI"])

        if self.__accions is None:
            start = time.perf_counter()
            e = EstatAdv(pos, adv, desti, parets, dim, True)
            f = self.minimax(e, float("-inf"), float("inf"))
            # agafar les accions de max (totes les parells del cami al node final)
            self.__accions = f.cami[::2]
            end = time.perf_counter()
            self.npases = len(self.__accions if self.__accions else [])
            # print(f"{self.nom} ha trobat solució en {end - start:.6f} segons ({self.npases} pases)")

        if self.__accions: return self.__accions.pop(0)
        return "ESPERAR", ""