import copy


class Estat:
    MAX_ANIMALS = 3

    accions = [(0, 1), (0, 2), (1, 0), (1, 1), (2, 0)] # Poll, Llop

    def __init__(self, local_barca: str, llops_esq: int, polls_esq: int, cami=None):
        if cami is None:
            cami = []

        self.llops_esq = llops_esq
        self.quica_esq = polls_esq
        self.local_barca = local_barca

        self.cami = cami

    def __hash__(self):
        return hash((self.llops_esq, self.quica_esq))

    @staticmethod
    def __canvi_posicio(lloc):
        if lloc == "ESQ":
            return "DRET"
        else:
            return "ESQ"


    @property
    def llops_dreta(self):
        return self.MAX_ANIMALS - self.llops_esq

    @property
    def quica_dreta(self):
        return self.MAX_ANIMALS - self.quica_esq

    def __eq__(self, other):
        """ Mètode per comparar dos estats.

        Args:
            other: Instància d'Estat a comparar.

        Returns:
            Booleà indicant si són iguals o no.
        """
        return (
                self.llops_esq == other.llops_esq
                and self.quica_esq == other.quica_esq
                and self.local_barca == other.local_barca
        )

    def es_meta(self) -> bool:
        """ Mètode per comprovar si s'ha arribat a l'estat meta.

        Returns:
            Booleà indicant si s'ha arribat a l'estat meta o no.
        """
        return self.quica_esq == 0 and self.llops_esq == 0

    def es_segur(self) -> bool:
        """ Únicament és segur si hi ha manco llops que gallines, o bé no hi ha gallines.

        Returns:
            Booleà indicant si és segur o no.
        """
        return (
                self.quica_esq >= self.llops_esq or self.quica_esq == 0
        ) and (
                self.quica_dreta >= self.llops_dreta or self.quica_dreta == 0
        )


    def transicio(self, accio):
        """ Mètode per realitzar la transició d'un estat a un altre.

        Args:
            accio: Tuple amb el nombre de polls i llops a moure.

        Returns:
            Tuple amb el nou estat i un booleà indicant si és legal o no.
        """
        nou_estat = copy.deepcopy(self)

        mov_poll, mov_llop = accio
        if mov_poll + mov_llop <= 2 :
            if self.local_barca == "ESQ":
                # Si la barca és a l'esquerra restam animals a aquella illa.
                mov_poll = -mov_poll
                mov_llop = -mov_llop

            nou_estat.quica_esq += mov_poll
            nou_estat.llops_esq += mov_llop

            nou_estat.local_barca = Estat.__canvi_posicio(self.local_barca)

        es_legal = (0 <= nou_estat.llops_esq <= Estat.MAX_ANIMALS) and (0 <= nou_estat.quica_esq <= Estat.MAX_ANIMALS)

        return nou_estat, es_legal

    def genera_fill(self) -> list:
        """ Mètode per generar els estats fills.

        Genera tots els estats fill legals a partir de l'estat actual.

        Returns:
            Llista d'estats fills generats.
        """
        estats_generats = []

        for moviments in self.accions:
            fill, es_legal = self.transicio(moviments)
            if es_legal:
                fill.cami.append(moviments)
                estats_generats.append(fill)

        return estats_generats

    def __str__(self):
        return (f"Llops esq: {self.llops_esq}, Quiques esq: {self.quica_esq} | "
                f"Llops dreta: {self.llops_dreta}, Quiques dreta: {self.quica_dreta} | Accio {self.cami}")