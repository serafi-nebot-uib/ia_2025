# Intel·ligència artificial (UIB)
### 2025-2026

En aquest repositori podreu trobar tots els exemples vists a classes.

## Instal·lació

Per poder emprar aquests exemples necessitem primerament tenir un entorn de ``Python`` amb les 
llibreries instal·lades. Per fer-ho s'han de seguir les següents passes. Emprarem ``uv``: 

**Primer pas**: 

Descarregar el repositori.
```
     git clone https://github.com/miquelmn/ia_2025
```

**Segon pas**:

Instal·lar [``uv``](https://docs.astral.sh/uv/), el gestor de paquets que emprarem.


**Tercer pas**

Activar l'entorn de ``python``
```
uv venv .venv
source .venv/bin/activate
uv pip install --editable
```

**Quart pas**

Comprovar que tot ha anat correctament amb la següent operació a la terminal

```
run-task prova
```

Si tot ha anat bé el resultat hauria de ser:

```
Tot ha funcionat
```

## Execució

Una vegada instal·lat per executar els diferents exemples vists a classe, des de la terminal, s'han d'emprar les 
següents instruccions:

```
source .venv/bin/activate
run-task <nom de la pràctica>
```

Una altra opció és fer el següent:

```
source .venv/bin/activate
PYTHONPATH=src python -m <nom de la pràctica> 
```

## Resum de les pràctiques

### Introducció i cerques
1. **Agents simples**: Introduïm un agent reflex simple i amb memòria per resoldre el problema de l'aspirador.
2. **Cerques no informades**: Aplicam la cerca en profunditat i en amplada per resoldre el problema de les gallines i els llops.
3. **Cerques informades**: Introduïm l'algoritme A* per resoldre el problema de les monedes.
4. **MINIMAX**: Aplicam l'algoritme de MINIMAX per resoldre el tres en línia.

### Aprenentatge per reforç
5. **Aprenentatge per reforç**: Problema d'aprenentatge per reforç.

### Aprenentatge automàtic
6. **Models lineals (I): Introducció ML i Perceptró**. Primera pràctica amb aprenentatge automàtic, emprant el perceptró.
7. **Models lineals (II): Regressió i correlació**: Explicam el problema de regressió i introduïm la correlació.
8. **Models lineals (III): Regressió logística i K-fold**: Introduïm la regressió logística i el K-Fold.
9. **SVM**. SVM lineal i ara podem triar els millors hiperparàmetres.
10. **Neteja de dades i models arboris**. Com es fa la neteja de dades i els darrers models de l'assignatura.