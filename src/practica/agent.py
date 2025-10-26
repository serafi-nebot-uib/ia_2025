from practica import joc
from practica.estat import Pos, Estat, EstatAdv
from queue import PriorityQueue
import time

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
        parets = {Pos(*p) for p in percepcio["PARETS"]}
        desti = Pos(*percepcio["DESTI"])

        if self.__accions is None:
            start = time.perf_counter()
            if adv is not None:
                e = EstatAdv(pos, adv, desti, parets, dim, True)
                f = self.minimax(e, float("-inf"), float("inf"))
                # agafar les accions de max (totes les parells del cami al node final)
                self.__accions = f.cami[::2]
            else:
                e = Estat(pos, desti, parets, dim)
                self.astar(e)
            end = time.perf_counter()
            print(f"{self.nom} ha trobat solució en {end - start:.6f} segons ({len(self.__accions if self.__accions else [])} pases)")

        if self.__accions:
            e = EstatAdv(pos, adv, desti, parets, dim, True) if adv is not None else Estat(pos, desti, parets, dim)
            accio = self.__accions.pop(0)
            # print(EstatAdv.tostr([e, e.accio(*accio)]))
            # print(accio)
            return accio

        return "ESPERAR", ""