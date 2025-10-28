from practica import joc
from practica.estat import Pos, Estat, EstatAdv
from queue import PriorityQueue

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
            # obtenir el darrer estat inserit -> cerca per profunditat
            estat_actual = oberts.pop(-1)

            if estat_actual is None: break
            if estat_actual in tancats: continue
            tancats.add(estat_actual)
            # h(n) == 0 -> estat final, no fa falta seguir cercant
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
            estat = Estat(pos, desti, parets, dim)
            self.dfs(estat)

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

        # podem passar Estat directament a PriorityQueue perque té el
        # mètode __lt__ implementat (necessari per a ordenar) prenent la funció f(n) = h(n) + c(n) com a valor
        oberts.put(estat_inicial)
        while oberts:
            estat_actual = oberts.get()

            if estat_actual is None: break
            if estat_actual in tancats: continue
            tancats.add(estat_actual)
            # h(n) == 0 -> estat final, no fa falta seguir cercant
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
            estat = Estat(pos, desti, parets, dim)
            self.astar(estat)

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
        # el mètode minimax() en molts de casos es crida moltes vegades amb el mateix estat
        # s'utilitza una cache, quan s'indica amb cache=True, per a evitar el cálcul d'estats ja calculats
        # en molts de casos disminueix bastant el temps de cerca però utilitza molta més memòria
        self.__cache = {} if cache else None

    def minimax(self, estat: EstatAdv, alpha: float | None = None, beta: float | None = None) -> EstatAdv:
        # s'utilitzen tots els arguments del mètode com a clau al diccionari de cache
        key = (estat, alpha, beta)
        if self.__cache and key in self.__cache: return self.__cache[key]

        self.estats.add(estat)
        self.nestats += 1

        # cas base (si és node fulla; no hi ha més moviments o algú ha guanyat)
        if estat.meta: return estat

        fills = []
        prune = None not in (alpha, beta)
        for fill in estat.fills:
            # obtenir el millor estat per al fill actual
            final = self.minimax(fill, alpha, beta)

            if prune:
                if estat.torn: alpha = max(alpha, final.puntuacio)
                else: beta = min(beta, final.puntuacio)

            fills.append(final)

            if prune and alpha >= beta: break

        # obtenir el millor estat finall de tots els fills
        fn = max if estat.torn else min
        best = fn(fills, key=lambda x: x.puntuacio)

        # emmagatzemar el resultat a la cache
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
            estat = EstatAdv(pos, adv, desti, parets, dim, True)
            alpha, beta = (float("-inf"), float("inf")) if self.__poda else (None, None)
            final = self.minimax(estat, alpha, beta) # obtenir l'estat final
            self.__accions = final.cami[::2] # agafar les accions de max (totes les accions parell del cami al node final)
            self.npases = len(self.__accions)

        if self.__accions: return self.__accions.pop(0)
        return "ESPERAR", ""