from typing import Self, NamedTuple
from collections.abc import Iterable
import operator as ops
from functools import cached_property

class Pos(NamedTuple):
    x: int
    y: int

    def __hash__(self): return hash((self.x, self.y))
    def __str__(self): return f"({self.x}, {self.y})"
    def __repr__(self): return f"({self.x}, {self.y})"

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

    def __init__(self, pos: Pos, desti: Pos, parets: set[Pos] | frozenset[Pos], dim: tuple[int, int], cami: list[tuple[str, str]] | None = None):
        self.pos, self.desti, self.parets, self.dim = pos, desti, frozenset(parets), dim
        self.cami = cami if cami is not None else []
        # cost: suma de totes les accions que s'han fet per a arribar a l'estat actual
        self.c = sum(self.ACCIO[a] for a, _ in self.cami)
        # heurística: distància manhattan entre la posició actual i el destí
        self.h = sum(abs(pos - desti))

    # una posició és vàlida per a accedir-hi si està dintre del tauler i no hi ha cap paret
    def _pos_valid(self, pos: Pos) -> bool: return (0, 0) <= pos < self.dim and pos not in self.parets

    def accio(self, accio: str, desp: str) -> Self | None:
        if accio not in self.ACCIO or desp not in self.DESP: raise KeyError(f"acció invàlida: {accio}, {desp}")
        # desplaçar la posició actual cap la direcció especificada, botant si s'ha especificat
        desti = self.pos + self.DESP[desp] * (1 + int(accio == "BOTAR"))
        if self._pos_valid(desti):
            # el set de parets s'han de copiar per a evitar que els estats fills modifiquin les parets dels pares i viceversa
            # copy() en lloc de deepcopy() per a millorar el rendiment
            pos, parets = self.pos, set(self.parets)
            if accio in ("MOURE", "BOTAR"):
                pos = desti
                parets.add(self.pos)
            else:
                if desti == self.desti: return None
                parets.add(desti)
            return self.__class__(pos, self.desti, parets, self.dim, self.cami + [(accio, desp)])

    @cached_property
    def fills(self) -> list[Self]:
        # obtenir els estats per a totes les acciones possible i per a tots els desplaçaments possibles
        # filter(None) per a llevar els estats invàlids que retorna accio()
        return [f for a in self.ACCIO.keys() for d in self.DESP.keys() if (f := self.accio(a, d))]

    # meta només comprova si l'estat actual és un node fulla (qualcú ha guanyat o no hi ha més accions possibles)
    @cached_property
    def meta(self) -> bool: return self.pos == self.desti or len(self.fills) == 0

    def __eq__(self, other):
        if not isinstance(other, Estat): return NotImplemented
        return self.pos == other.pos and self.desti == other.desti and self.dim == other.dim

    def __lt__(self, other):
        if not isinstance(other, Estat): return NotImplemented
        sval, oval = self.h + self.c, other.h + other.c
        return self.h < other.h if sval == oval else sval < oval

    def __hash__(self): return hash((tuple(self.pos), tuple(self.desti), self.dim))

    def __str__(self): return Estat.tostr(self)

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

    @classmethod
    def tostr(cls, estats: Self | list, *, icons: dict[str, str] = ICONS) -> str:
        if estats is None: return ""
        if not isinstance(estats, list): estats = [estats]
        estats = list(filter(None, estats))
        if len(set(e.dim for e in estats)) != 1: return ""

        sep = estats[0].dim[0] * 2
        header = ""
        for e in estats:
            a, d = e.cami[-1] if e.cami else ("ESPERAR", None)
            s = " ".join(map(str, (a[0], d)))
            header += f"{s:<{sep}s}  "
        header += "\n"

        width, height = estats[0].dim
        n = len(estats)

        s = [[[icons["empty"] for _ in range(width)] for _ in range(height)] for _ in range(n)]
        for board, estat in zip(s, estats):
            board[estat.pos[1]][estat.pos[0]] = icons["robot"]
            board[estat.desti[1]][estat.desti[0]] = icons["goal"]
            for p in estat.parets: board[p[1]][p[0]] = icons["wall"]
        return header + "\n".join("  ".join("".join(c) for c in b) for b in zip(*s))

class EstatAdv:
    # totes les accions possibles
    ACCIO = { "MOURE", "BOTAR", "POSAR_PARET" }
    # tots els desplaçaments possibles amb la seva direcció (x, y)
    DESP = { "N": Pos(0, -1), "O": Pos(-1, 0), "S": Pos(0, 1), "E": Pos(1, 0) }

    def __init__(self, pos: Pos, adv: Pos, desti: Pos, parets: set[Pos] | frozenset[Pos], dim: tuple[int, int], torn: bool = True, cami: list[tuple[str, str]] | None = None):
        self.pos, self.adv, self.desti, self.parets, self.dim = pos, adv, desti, frozenset(parets), dim
        self.torn = torn # True si el torn actual es de l'agent on esta pos, False si ho es de adv
        self.cami = cami if cami is not None else []
        # distància manhattan entre la posició de cada agent i el destí
        self.dist = sum(abs(pos - desti))
        self.dist_adv = sum(abs(adv - desti))

    # una posició és vàlida per a accedir-hi si està dintre del tauler, no hi ha cap paret i tampoc hi ha l'adversari
    def _pos_valid(self, pos: Pos, adv: Pos) -> bool: return (0, 0) <= pos < self.dim and pos not in self.parets and pos != adv

    def accio(self, accio: str, desp: str) -> Self | None:
        if accio not in self.ACCIO or desp not in self.DESP: raise KeyError(f"acció invàlida: {accio}, {desp}")
        # EstatAdv manté constància del torn actual, per tant s'ha de tenir en compte si la posició inicial és de min o max
        pos, adv = self.pos, self.adv
        pos_inicial, pos_adv = (pos, adv) if self.torn else (adv, pos)
        # desplaçar la posició actual cap la direcció especificada, botant si s'ha especificat
        desti = pos_inicial + self.DESP[desp] * (1 + (accio == "BOTAR"))
        if self._pos_valid(desti, pos_adv):
            pos, parets = self.pos, set(self.parets)
            if accio in ("MOURE", "BOTAR"):
                if self.torn: pos = desti
                else: adv = desti
                parets.add(pos_inicial)
            else:
                if desti == self.desti: return None
                parets.add(desti)
            return self.__class__(pos, adv, self.desti, parets, self.dim, not self.torn, self.cami + [(accio, desp)])

    @cached_property
    def fills(self) -> list[Self]:
        # obtenir els estats per a totes les acciones possible i per a tots els desplaçaments possibles
        # filter(None) per a llevar els estats invàlids que retorna accio()
        fills = [f for a in self.ACCIO for d in self.DESP.keys() if (f := self.accio(a, d))]
        # en cas de que ja no hi hagi més moviments possibles
        # s'afegeix l'acció "ESPERAR" per a permetre que l'agent contrari pugui seguir fent moviments (sinó l'arbre d'accions acaba aquí)
        # també es comprova que l'acció anterior (la del contrincant) no sigui "ESPERAR" per a evitar un arbre infinit
        if len(fills) == 0:
            if not self.cami or self.cami[-1][0] != "ESPERAR":
                e = self.__class__(self.pos, self.adv, self.desti, self.parets, self.dim, not self.torn, self.cami + [("ESPERAR", "")])
                fills.append(e)
        return fills

    # meta només comprova si l'estat actual és un node fulla (qualcú ha guanyat o no hi ha més accions possibles)
    @cached_property
    def meta(self) -> bool: return self.desti in (self.pos, self.adv) or len(self.fills) == 0

    @cached_property
    def puntuacio(self):
        if self.dist == 0: return self.dist_adv
        if self.dist_adv == 0: return -self.dist
        return 0

    def __eq__(self, other):
        if not isinstance(other, EstatAdv): return NotImplemented
        return (
            self.pos == other.pos and
            self.adv == other.adv and
            self.desti == other.desti and
            self.dim == other.dim and
            self.torn == other.torn and
            self.parets == other.parets
        )

    def __hash__(self): return hash((tuple(self.pos), tuple(self.adv), tuple(self.desti), self.dim, self.torn, self.parets))

    def __str__(self): return EstatAdv.tostr(self)

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

    @classmethod
    def tostr(cls, estats: Self | list, *, icons: dict[str, str] = ICONS) -> str:
        if estats is None: return ""
        if not isinstance(estats, list): estats = [estats]
        estats = list(filter(None, estats))
        if len(set(e.dim for e in estats)) != 1: return ""

        header = ""
        sep = estats[0].dim[0] * 2
        header += "  ".join(f"{'max' if e.torn else 'min':<{sep}s}" for e in estats) + "\n"
        header += "  ".join(f"{e.puntuacio:<{sep}d}" for e in estats) + "\n"
        for e in estats:
            a, d = e.cami[-1] if e.cami else ("ESPERAR", None)
            s = " ".join(map(str, (a[0], d)))
            header += f"{s:<{sep}s}  "
        header += "\n"

        width, height = estats[0].dim
        n = len(estats)

        s = [[[icons["empty"] for _ in range(width)] for _ in range(height)] for _ in range(n)]
        for board, estat in zip(s, estats):
            board[estat.pos[1]][estat.pos[0]] = icons["robot"]
            board[estat.adv[1]][estat.adv[0]] = icons["enemy"]
            board[estat.desti[1]][estat.desti[0]] = icons["goal"]
            for p in estat.parets: board[p[1]][p[0]] = icons["wall"]
        return header + "\n".join("  ".join("".join(c) for c in b) for b in zip(*s))