from typing import NamedTuple, Self
from collections.abc import Iterable
import operator as ops

class Pos(NamedTuple):
    x: int
    y: int

    def within(self, a: Iterable[int], b: Iterable[int] | None = None) -> bool:
        """
        Comprovar si Pos compleix `a <= pos < b` element per element.
        Si no es passa l'argument `b` es comprova `(0, 0) <= pos < a`
        """
        if b is None: b, a = a, (0,) * len(self)
        return a <= self < b

    # conjunt d'operacions aritmètiques i lògiques amb altres objectes Pos, int o qualsevol Iterable
    # les operacións aritmètiques retornen un objecte Pos nou amb el resultat
    # les operacións lògiques retornen un booleà (en concret comproven que tots els elements de Pos compleixen la condició lògica)
    def _op_unary(self, op): return (op(a) for a in self)
    def _op_binary(self, other, op):
        match other:
            case int():      return (op(c, other) for c in self)
            case Iterable(): return (op(a, b) for a, b in zip(self, other))
            case _:          return NotImplemented

    def __pos__(self):             return Pos(*self._op_unary(ops.pos))
    def __neg__(self):             return Pos(*self._op_unary(ops.neg))
    def __abs__(self):             return Pos(*self._op_unary(ops.abs))
    def __add__(self, other):      return Pos(*self._op_binary(other, ops.add))
    def __sub__(self, other):      return Pos(*self._op_binary(other, ops.sub))
    def __mul__(self, other):      return Pos(*self._op_binary(other, ops.mul))
    def __mod__(self, other):      return Pos(*self._op_binary(other, ops.mod))
    def __pow__(self, other):      return Pos(*self._op_binary(other, ops.pow))
    def __truediv__(self, other):  return Pos(*self._op_binary(other, ops.truediv))
    def __iadd__(self, other):     return Pos(*self._op_binary(other, ops.add))
    def __isub__(self, other):     return Pos(*self._op_binary(other, ops.sub))
    def __imul__(self, other):     return Pos(*self._op_binary(other, ops.mul))
    def __imod__(self, other):     return Pos(*self._op_binary(other, ops.mod))
    def __ipow__(self, other):     return Pos(*self._op_binary(other, ops.pow))
    def __itruediv__(self, other): return Pos(*self._op_binary(other, ops.truediv))
    def __lt__(self, other):       return all(self._op_binary(other, ops.lt))
    def __le__(self, other):       return all(self._op_binary(other, ops.le))
    def __eq__(self, other):       return all(self._op_binary(other, ops.eq))
    def __ne__(self, other):       return all(self._op_binary(other, ops.ne))
    def __ge__(self, other):       return all(self._op_binary(other, ops.ge))
    def __gt__(self, other):       return all(self._op_binary(other, ops.gt))

    def __hash__(self): return hash(tuple(self))
    def __str__(self): return str(tuple(self))

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

    def accio(self, accio: str, desp: str) -> Self | None:
        if desp not in self.DESP: raise KeyError(f"desplaçament invalid: {desp}")
        pos, parets = self.pos, self.parets
        match accio:
            case "MOURE":       pos += self.DESP[desp]
            case "BOTAR":       pos += self.DESP[desp] * 2
            case "POSAR_PARET": parets = self.parets.union({pos})
            case _:             raise KeyError(f"accio invalida: {accio}")
        if pos.within(self.dim) and pos not in self.parets:
            return self.__class__(pos, self.desti, parets, self.dim, self.cami + [(accio, desp)])

    def fills(self) -> list[Self]:
        # obtenir els estats per a totes les acciones possible i per a tots els desplaçaments possibles
        # filter(None) per a llevar els estats invàlids que retorna accio()
        return list(filter(None, (self.accio(a, d) for a in self.ACCIO.keys() for d in self.DESP.keys())))

    def __eq__(self, other):
        if not isinstance(other, Estat): return NotImplemented
        g = ops.attrgetter("pos", "desti", "parets", "dim")
        return g(self) == g(other)

    def __lt__(self, other):
        if not isinstance(other, Estat): return NotImplemented
        sval, oval = self.h + self.c, other.h + other.c
        return self.h < other.h if sval == oval else sval < oval

    def __hash__(self):
        # parets s'ordena abans per a assegurar que hash sempre sigui el mateix per al mateix conjunt de parets
        # (python no assegura que dos set() amb els mateixos elements seguesquin el mateix ordre)
        return hash(self.pos + self.desti + self.dim + tuple(b for a in sorted(self.parets) for b in a))