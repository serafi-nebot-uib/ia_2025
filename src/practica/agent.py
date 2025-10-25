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

    def minimax(self, estat: EstatAdv, alpha: float, beta: float) -> EstatAdv:
        if estat.meta: return estat

        fills = []
        for f in estat.fills:
            v = self.minimax(f, alpha, beta)
            if f.torn: alpha = max(alpha, v.value)
            else: beta = min(beta, v.value)
            fills.append(v)
            if alpha >= beta: break

        fn = max if estat.torn else min
        return fn(fills, key=lambda x: x.value)

    def actua(self, percepcio) -> tuple[str, str]:
        if self.__accions is None:
            dim = percepcio["MIDA"]
            torn = percepcio["TORN"]
            pos = Pos(*percepcio["AGENTS"][torn])
            adv = next((Pos(*v) for k, v in percepcio["AGENTS"].items() if k != torn), None)
            parets = {Pos(*p) for p in percepcio["PARETS"]}
            desti = Pos(*percepcio["DESTI"])

            if adv is not None:
                e = EstatAdv(pos, adv, desti, parets, dim, True)

                start = time.perf_counter()
                f = self.minimax(e, float("-inf"), float("inf"))
                end = time.perf_counter()
                print(f"minimax in {end - start:.6f} seconds")

                # agafar les accions de max (totes les parells del cami al node final)
                self.__accions = f.cami[::2]

            print(f"n steps: {len(self.__accions if self.__accions else [])}")

        if self.__accions:
            return self.__accions.pop(0)

        return "ESPERAR", ""