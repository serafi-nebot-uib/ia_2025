from typing import NamedTuple, Self, Iterable
import operator as ops

class Pos(NamedTuple):
    x: int; y: int
    def within(self, low: Iterable[int], up: Iterable[int] | None = None) -> bool:
        if up is None: up, low = low, (0, 0)
        return all(l <= s < u for l, s, u in zip(low, self, up))
    def __op(self, other, op):
        if isinstance(other, int): return Pos(*(op(c, other) for c in self))
        elif isinstance(other, Iterable): return Pos(*(op(a, b) for a, b in zip(self, other)))
        else: return NotImplemented
    def __add__(self, other): return self.__op(other, ops.add)
    def __sub__(self, other): return self.__op(other, ops.sub)
    def __mul__(self, other): return self.__op(other, ops.mul)
    def __iadd__(self, other): return self.__op(other, ops.add)
    def __isub__(self, other): return self.__op(other, ops.sub)
    def __imul__(self, other): return self.__op(other, ops.mul)
    def __neg__(self): return Pos(-self.x, -self.y)
    def __str__(self): return str((self.x, self.y))
    def __repr__(self): return str(self)

class Estat:
    ACCIO = { "MOURE": 1, "BOTAR": 2, "POSAR_PARET": 3 }
    DESP = { "N": Pos(0, -1), "O": Pos(-1, 0), "S": Pos(0, 1), "E": Pos(1, 0) }

    def __init__(self, pos: Pos, desti: Pos, parets: set[Pos], dim: tuple[int, int], cami: list[tuple[str, str]] | None = None):
        self.pos, self.desti, self.parets, self.dim = pos, desti, parets, dim
        self.cami = cami if cami is not None else []
        self.__h = sum(abs(a - b) for a, b in zip(pos, desti))

    @property
    def h(self) -> int: return self.__h
    @property
    def c(self) -> int: return sum(self.ACCIO[a] for a, _ in self.cami)

    def accio(self, accio: str, desp: str) -> Self | None:
        if desp not in self.DESP: raise KeyError(f"desplaçament invalid: {desp}")
        cami = self.cami + [(accio, desp)]
        pos = self.pos + self.DESP[desp]
        if accio == "MOURE":
            if pos.within(self.dim) and pos not in self.parets:
                return self.__class__(pos, self.desti, self.parets, self.dim, cami)
        elif accio == "POSAR_PARET":
            if pos.within(self.dim) and pos not in self.parets:
                return self.__class__(self.pos, self.desti, self.parets.union({pos}), self.dim, cami)
        elif accio == "BOTAR":
            pos, posb = pos + self.DESP[desp], pos
            if all(p.within(self.dim) and p not in self.parets for p in (pos, posb)):
                return self.__class__(pos, self.desti, self.parets, self.dim, cami)
        else:
            raise KeyError(f"accio invalida: {accio}")
        return None

    def fills(self): yield from filter(None, (self.accio(a, d) for a in self.ACCIO.keys() for d in self.DESP.keys()))

    def __eq__(self, other):
        if not isinstance(other, Estat): return NotImplemented
        return all(getattr(self, k) == getattr(other, k) for k in ("pos", "desti", "parets", "dim"))
    def __lt__(self, other):
        if not isinstance(other, Estat): return NotImplemented
        sval, oval = self.h + self.c, other.h + other.c
        return self.h < other.h if sval == oval else sval < oval
    def __hash__(self):
        # parets s'ordena abans de ser aplanat per a assegurar que hash sempre suigui el mateix per al mateix conjunt de parets
        # (python no assegura dos set() amb els mateixos elements seguesquin el mateix ordre)
        return hash(self.pos + self.desti + self.dim + tuple(b for a in sorted(self.parets) for b in a))
    def __str__(self): return f"{'x'.join(map(str, self.dim))} | {len(self.parets):3d} : {self.pos} -> {self.desti}"
    def __repr__(self): return str(self)