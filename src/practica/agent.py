from practica import joc
from practica.estat import Pos, Estat
from queue import PriorityQueue

MODE = 2  # 0:A*, 1:DFS, 2:MINIMAX

class Viatger(joc.Viatger):
    def __init__(self, *args, **kwargs):
        super(Viatger, self).__init__(*args, **kwargs)
        self.__accions = None
        self.__tancats = None # Canviar tancats() per self.__tancats a astar i dfs per evitar emprar dues variables
        #self.__visitats = None

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
            if estat_actual.h == 0: break # Si la distància entre la posicio actual i el desti es 0 vol dir que hem arribat a la meta

            for f in estat_actual.fills(): oberts.append(f)
            tancats.add(estat_actual)

        # Si estat_actual es un objecte (no es NONE) i la distància entre la posicio actual i el desti es 0 
        # vol dir que hem arribat a la meta
        if estat_actual and estat_actual.h == 0: 
            self.__accions = estat_actual.cami
            exit = True

        return exit


    def minimax(self, estat: Estat, alpha, beta, torn_max=True, visitats=None) -> tuple[Estat, int]:
        if visitats is None:
            visitats = set()

        # A minimax la puntuació es calcula avaluant quin dels dos agents te la millor heuristica
        ###
        # S'heurística es pot calcul·lar de formes més complexes que tenguin en compte si l'agent està bloquejat o no
        # ja que si l'heurística és la distància Manhattan posar parets mai influenciarà a la puntuació de l'agent contrari
        # ni beneficiarà a l'agent MAX i mai posarà paret, lo qual no té per què estar malament, simplement no posarà mai parets
        ###

        if (estat and estat.h == 0) : return estat.resultat() #
            
        visitats.add(estat)
        puntuacio_fills = []

        print(f"\n_____ Inici minimax _____\n       TORN DE {"MAX" if torn_max else "MIN"}\n")
        fills_generats = estat.fills()
        print(f"Estat Pare: (h= {estat.h}, hash: {estat.__hash__()})\n{estat}")
        print(f"Estat Fill generats:\n{Estat.inline_print(fills_generats)}")
        print("\n".join(f"{i}: (h={v.h}, hash={hash(v)})" for i, v in enumerate(fills_generats))+"\n")
        for fill in fills_generats:
            if fill in visitats: # if (fill == estat) or (fill in visitats) :
                continue
            punt_fill = None
            if fill not in self.__tancats:
                print(f"Estat Pare: (h= {estat.h}, hash: {hash(estat)})\n{estat}")
                print(f"Estat Fill: (h= {fill.h}, hash: {hash(fill)})\n{fill}")
                #print("this estat fill is not in self.__tancats\nself.__tancats\n" +"\n".join(f"({hash(k)}, {v[1]})" for k, v in self.__tancats.items()) + "\n")
                punt_fill = self.minimax(fill, alpha, beta, not torn_max, visitats.copy()) # Rep (estat, puntuacio)
                print(f"punt_fill: ({hash(punt_fill[0])}, {punt_fill[1]})")
                
                if torn_max:
                    alpha = max(alpha, punt_fill[1])
                else:
                    beta = min(beta, punt_fill[1])

                print(f"α={alpha}, β={beta}\n")
                if alpha >= beta:
                    puntuacio_fills.append(punt_fill) 
                    break

                self.__tancats[fill] = punt_fill
                #print(f"Added estat fill to self.__tancats:\n{ [(hash(k), v[1]) for k, v in self.__tancats.items()] }\n")
            if punt_fill is not None : puntuacio_fills.append(punt_fill) # Afegeix cada fill a la llista, un fill es una tupla de (Estat, distància al desti)
        
        if len(puntuacio_fills) == 0:
            #print("puntuacio_fills is empty, returning current estat")
            return estat.resultat()

        puntuacio_fills = sorted(puntuacio_fills, key=lambda x: x[1]) # Ordena els fills en ordre ascendent per distància respecte al destí
        print(f"Content of puntuacio_fills = {[ (hash(estat), puntuacio) for estat, puntuacio in puntuacio_fills ]}")
        print(f"Torn de {'MAX' if torn_max else 'MIN'}, returning puntuacio_fills[{0 if torn_max else -1}] = ({hash(puntuacio_fills[0 if torn_max else -1][0])}, {puntuacio_fills[0 if torn_max else -1][1]})\n")
        if torn_max:
            return puntuacio_fills[0]
        else:
            return puntuacio_fills[-1]
        


    def actua(self, percepcio) -> tuple[str, str]:
        if self.__accions is None:
            dim = percepcio["MIDA"]
            pos = Pos(*percepcio["AGENTS"][percepcio["TORN"]]) # Objecte Pos amb les coordenades x,y on es troba l'agent del torn actual
            parets = {Pos(*p) for p in percepcio["PARETS"]} # Crea un conjunt (set) de objectes Pos, cada Pos indica una paret amb el seu (posX,posY)
            desti = Pos(*percepcio["DESTI"]) # Objecte Pos amb les coordenades de destí

            match MODE:
                case 0: self.astar(Estat(pos, desti, parets, dim))
                case 1: self.dfs(Estat(pos, desti, parets, dim))
                case 2:
                    print(f"Torn de l'{percepcio["TORN"]} 🤖")
                    pos_adv = next(Pos(*v) for k, v in percepcio["AGENTS"].items() if k != percepcio["TORN"])
                    print(f"pos:{pos}\npos_adv:{pos_adv}")

                    self.__tancats = dict() # set()
                    estat, _ = self.minimax(Estat(pos, desti, parets, dim, cami=None, pos_adv=pos_adv), alpha=-float('inf'), beta=float('inf'))            
                    self.__accions = estat.cami

            print(f"n steps: {len(self.__accions if self.__accions else [])}")
        if self.__accions:
            accio = self.__accions.pop(0)
            print(accio)
            return accio
        return ("ESPERAR", "")