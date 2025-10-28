from practica import joc
from practica.estat import Pos, Estat, EstatAdv
from queue import PriorityQueue
import time

class ViatgerDFS(joc.Viatger):
    def __init__(self, *args, **kwargs):
        super(ViatgerDFS, self).__init__(*args, **kwargs)
        self.__accions = None
        self.npases = 0
        self.nestats = 0

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
            self.npases = len(self.__accions)
            self.nestats = len(tancats)
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
            print(f"{self.nom} ha trobat solució en {end - start:.6f} segons ({self.npases} pases)")

            import sys
            sys.exit(0)

        if self.__accions: return self.__accions.pop(0)
        return "ESPERAR", ""

class ViatgerAstar(joc.Viatger):
    def __init__(self, *args, **kwargs):
        super(ViatgerAstar, self).__init__(*args, **kwargs)
        self.__accions = None
        self.npases = 0
        self.nestats = 0

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
            self.npases = len(self.__accions)
            self.nestats = len(tancats)
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
            # print(f"{self.nom} ha trobat solució en {end - start:.6f} segons ({self.npases} pases)")

        if self.__accions: return self.__accions.pop(0)
        return "ESPERAR", ""

class ViatgerMinimax(joc.Viatger):
    def __init__(self, poda: bool = True, cache: bool = False):
        super(ViatgerMinimax, self).__init__()
        self.__poda = poda
        self.__accions = None
        self.npases = 0
        self.nestats = 0
        self.estats = set()
        self.__cache = {} if cache else None

    def minimax(self, estat: EstatAdv, alpha: float | None = None, beta: float | None = None) -> EstatAdv:
        key = (estat, alpha, beta)
        if self.__cache and key in self.__cache: return self.__cache[key]

        self.estats.add(estat)
        self.nestats += 1
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
        best = fn(fills, key=lambda x: x.puntuacio)
        if self.__cache: self.__cache[key] = best
        return best

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
            alpha, beta = (float("-inf"), float("inf")) if self.__poda else (None, None)
            f = self.minimax(e, alpha, beta)
            self.__accions = f.cami[::2] # agafar les accions de max (totes les accions parell del cami al node final)
            self.npases = len(self.__accions)
            end = time.perf_counter()
            print(f"{self.nom} ha trobat solució en {end - start:.6f} segons ({self.npases} pases; {self.nestats} nestats; {len(self.estats)} estats)")

        if self.__accions: return self.__accions.pop(0)
        return "ESPERAR", ""