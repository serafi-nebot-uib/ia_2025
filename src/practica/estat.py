from typing import NamedTuple, Self
from collections.abc import Iterable
import operator as ops
from functools import cached_property
from copy import deepcopy

class Pos(NamedTuple):
    x: int
    y: int

    def __hash__(self): return hash(tuple(self))
    def __str__(self): return str(tuple(self))

    # conjunt d'operacions aritmètiques i lògiques amb altres objectes Pos, int o qualsevol Iterable
    # les operacións aritmètiques retornen un objecte Pos nou amb el resultat
    # les operacións lògiques retornen un booleà (en concret comproven que tots els elements de Pos compleixen la condició lògica)
    def _op_unary(self, op): return (op(a) for a in self)
    def _op_binary(self, op, other):
        match other:
            case int():      return (op(c, other) for c in self)
            case Iterable(): return (op(a, b) for a, b in zip(self, other))
            case _:          return NotImplemented

    def __pos__(self):             return Pos(*self._op_unary(ops.pos))
    def __neg__(self):             return Pos(*self._op_unary(ops.neg))
    def __abs__(self):             return Pos(*self._op_unary(ops.abs))
    def __add__(self, other):      return Pos(*self._op_binary(ops.add, other))
    def __sub__(self, other):      return Pos(*self._op_binary(ops.sub, other))
    def __mul__(self, other):      return Pos(*self._op_binary(ops.mul, other))
    def __mod__(self, other):      return Pos(*self._op_binary(ops.mod, other))
    def __pow__(self, other):      return Pos(*self._op_binary(ops.pow, other))
    def __truediv__(self, other):  return Pos(*self._op_binary(ops.truediv, other))
    def __lt__(self, other):       return all(self._op_binary(ops.lt, other))
    def __le__(self, other):       return all(self._op_binary(ops.le, other))
    def __eq__(self, other):       return all(self._op_binary(ops.eq, other))
    def __ge__(self, other):       return all(self._op_binary(ops.ge, other))
    def __gt__(self, other):       return all(self._op_binary(ops.gt, other))

class Estat:
    # totes les accions possibles amb el seu cost associat
    ACCIO = { "MOURE": 1, "BOTAR": 2, "POSAR_PARET": 3 }
    # tots els desplaçaments possibles amb la seva direcció (x, y)
    DESP = { "N": Pos(0, -1), "O": Pos(-1, 0), "S": Pos(0, 1), "E": Pos(1, 0) }

    def __init__(self, pos: Pos, desti: Pos, parets: set[Pos], dim: tuple[int, int], cami: list[tuple[str, str]] | None = None):
        self.pos, self.desti, self.parets, self.dim = pos, desti, parets, dim
        self.cami = cami if cami is not None else []
        # cost: suma de totes les accions que s'han fet per a arribar a l'estat actual
        self.c = sum(self.ACCIO[a] for a, _ in self.cami)
        # heurística: distància manhattan entre la posició actual i el destí
        self.h = sum(abs(pos - desti))

    def _pos_valid(self, pos: Pos) -> bool: return (0, 0) <= pos < self.dim and pos not in self.parets

    def accio(self, accio: str, desp: str) -> Self | None:
        if accio not in self.ACCIO or desp not in self.DESP: raise KeyError(f"acció invàlida: {accio}, {desp}")
        desti = self.pos + self.DESP[desp] * (1 + (accio == "BOTAR"))
        if self._pos_valid(desti):
            pos, parets = self.pos, deepcopy(self.parets)
            match accio:
                case "MOURE" | "BOTAR":
                    pos = desti
                    parets.add(self.pos)
                case "POSAR_PARET":
                    if desti in self.desti: return None
                    parets.add(desti)
            return self.__class__(pos, self.desti, parets, self.dim, self.cami + [(accio, desp)])

    def fills(self) -> list[Self]:
        # obtenir els estats per a totes les acciones possible i per a tots els desplaçaments possibles
        # filter(None) per a llevar els estats invàlids que retorna accio()
        return list(filter(None, (self.accio(a, d) for a in self.ACCIO.keys() for d in self.DESP.keys())))

    def __eq__(self, other):
        if not isinstance(other, Estat): return NotImplemented
        return hash(self) == hash(other)

    def __lt__(self, other):
        if not isinstance(other, Estat): return NotImplemented
        sval, oval = self.h + self.c, other.h + other.c
        return self.h < other.h if sval == oval else sval < oval

    def __hash__(self):
        # parets s'ordena abans per a assegurar que hash sempre sigui el mateix per al mateix conjunt de parets
        # (python no assegura que dos set() amb els mateixos elements seguesquin el mateix ordre)
        return hash(self.pos + self.desti + self.dim + tuple(b for a in sorted(self.parets) for b in a))

class EstatAdv:
    # totes les accions possibles amb el seu cost associat
    ACCIO = { "MOURE": 1, "BOTAR": 2, "POSAR_PARET": 3 }
    # tots els desplaçaments possibles amb la seva direcció (x, y)
    DESP = { "N": Pos(0, -1), "O": Pos(-1, 0), "S": Pos(0, 1), "E": Pos(1, 0) }

    def __init__(self, pos: Pos, adv: Pos, desti: Pos, parets: set[Pos], dim: tuple[int, int], torn: bool = True, cami: list[tuple[str, str]] | None = None):
        self.pos, self.adv, self.desti, self.parets, self.dim = pos, adv, desti, parets, dim
        self.torn = torn # True si el torn actual es de l'agent on esta pos, False si ho es de adv
        self.cami = cami if cami is not None else []
        # cost: suma de totes les accions que s'han fet per a arribar a l'estat actual
        self.c = sum(self.ACCIO[a] for a, _ in self.cami)
        # heurística: distància manhattan entre la posició actual i el destí
        self.h = sum(abs(pos - desti))
        self.h_adv = sum(abs(adv - desti))

    def _pos_valid(self, pos: Pos, adv: Pos) -> bool: return (0, 0) <= pos < self.dim and pos not in self.parets and pos != adv

    def accio(self, accio: str, desp: str) -> Self | None:
        if accio not in self.ACCIO or desp not in self.DESP: raise KeyError(f"acció invàlida: {accio}, {desp}")
        pos, adv = self.pos, self.adv
        pos_inicial = pos if self.torn else adv
        pos_adv = adv if self.torn else pos
        desti = pos_inicial + self.DESP[desp] * (1 + (accio == "BOTAR"))
        if self._pos_valid(desti, pos_adv):
            pos, parets = self.pos, deepcopy(self.parets)
            match accio:
                case "MOURE" | "BOTAR":
                    if self.torn: pos = desti
                    else: adv = desti
                    parets.add(pos_inicial)
                case "POSAR_PARET":
                    if desti in self.desti: return None
                    parets.add(desti)
            return self.__class__(pos, adv, self.desti, parets, self.dim, not self.torn, self.cami + [(accio, desp)])

    @cached_property
    def fills(self) -> list[Self]:
        # obtenir els estats per a totes les acciones possible i per a tots els desplaçaments possibles
        # filter(None) per a llevar els estats invàlids que retorna accio()
        return [f for a in self.ACCIO.keys() for d in self.DESP.keys() if (f := self.accio(a, d))]

    # meta només comprova si l'estat actual és un node fulla (qualcú ha guanyat o no hi ha més accions possibles)
    @cached_property
    def meta(self) -> bool: return self.desti in (self.pos, self.adv) or len(self.fills) == 0

    @cached_property
    def value(self):
        if self.meta:
            if self.h == 0: return 1
            if self.h_adv == 0: return -1
            return 0
        fvals = (fill.value for fill in self.fills)
        return max(fvals) if self.torn else min(fvals)

    def __eq__(self, other):
        if not isinstance(other, Estat): return NotImplemented
        return hash(self) == hash(other)

    def __hash__(self):
        # parets s'ordena abans per a assegurar que hash sempre sigui el mateix per al mateix conjunt de parets
        # (python no assegura que dos set() amb els mateixos elements seguesquin el mateix ordre)
        return hash(map(tuple, (self.pos, self.adv, self.desti, self.dim, (self.torn,), (b for a in sorted(self.parets) for b in a))))
        # return hash(self.pos + self.adv + self.desti + self.dim + tuple(b for a in sorted(self.parets) for b in a))

    def __str__(self):
        icons = {
            "wall":  "██",
            "empty": "  ",
            "robot": "rr",
            "enemy": "ee",
            "goal":  "xx"
        }
        return EstatAdv.tostr(self)

    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_WHITE = "\033[47m"
    BG_BLUE = "\033[44m"
    RST = "\033[0m"

    ICONS = {
        "wall":  BG_BLACK + "  " + RST,
        "empty": BG_WHITE + "  " + RST,
        "robot": BG_BLUE  + "  " + RST,
        "enemy": BG_RED   + "  " + RST,
        "goal":  BG_GREEN + "  " + RST
    }

    @staticmethod
    def tostr(estats: Self | list, *, icons: dict[str, str] = ICONS) -> str:
        if estats is None: return ""
        if not isinstance(estats, list): estats = [estats]
        if len(set(e.dim for e in estats)) != 1: return ""

        width = estats[0].dim[0] * 2
        print("  ".join(f"{e.value:<{width}d}" for e in estats))
        print("  ".join(f"{str(e.meta):<{width}s}" for e in estats))
        print("  ".join(f"{str(e.torn):<{width}s}" for e in estats))
        for e in estats:
            a, d = e.cami[-1] if e.cami else ("ESPERAR", None)
            s = " ".join(map(str, (a[0], d)))
            print(f"{s:<{width}s}", end="  ")
        print()

        width, height = estats[0].dim
        n = len(estats)

        s = [[[icons["empty"] for _ in range(width)] for _ in range(height)] for _ in range(n)]
        for board, estat in zip(s, estats):
            board[estat.pos[1]][estat.pos[0]] = icons["robot"]
            board[estat.adv[1]][estat.adv[0]] = icons["enemy"]
            board[estat.desti[1]][estat.desti[0]] = icons["goal"]
            for p in estat.parets: board[p[1]][p[0]] = icons["wall"]
        return "\n".join("  ".join("".join(c) for c in b) for b in zip(*s))

        # NOTE: no se perque amb emoticonos no se me mostra be per pantalla, me va molt millor amb colors ANSI
        # wall, empt, goal, robot, enemy, board = "🟥", "🔲", "🏁", "🤖", "🏴‍☠️", "" #🛑
        # for j in range(estats[0].dim[0]):
        #     for brd_num in range(len(estats)):
        #         estat_act = estats[brd_num]
        #         for i in range(estat_act.dim[1]):
        #             if estat_act.pos == Pos(i, j):
        #                 board += robot
        #             elif estat_act.desti == Pos(i,j):
        #                 board += goal
        #             elif Pos(i,j) in estat_act.parets:
        #                 board += wall
        #             elif Pos(i,j) == estat_act.adv:
        #                 board += enemy
        #             else:
        #                 board += empt
        #         if brd_num < len(estats) - 1: board += "  "
        #     board += "\n"
        # return board