# Intel·ligència artificial - UIB (2025-2026)

En aquest repositori trobareu tots els exemples pràctics treballats durant les classes de l’assignatura d’Intel·ligència Artificial (21722).
## Instal·lació

Per poder emprar aquests exemples necessitem primerament tenir un entorn de ``Python`` amb les 
llibreries instal·lades. Per fer-ho s'han de seguir les següents passes. Emprarem ``uv``: 

### 1. Clonar el repositori 

Descarregar el repositori.
```
     git clone https://github.com/miquelmn/ia_2025
```

### 2. Instal·lar `uv`:

Instal·lar [``uv``](https://docs.astral.sh/uv/), el gestor de paquets que emprarem.


### 3. Crear i activar l'entorn

```
uv venv .venv
source .venv/bin/activate
uv pip install --editable .
```

### 4. Verificar que tot funciona correctament

Executau la següent comanda:

```
run-task prova
```

Si tot està ben configurat, haureu de veure el següent missatge:

```
Tot ha funcionat
```

## Execució

Un cop l'entorn està activat, podeu executar qualsevol pràctica amb una de les dues opcions següents:

### Opció 1: Emprant `run-task`

```
source .venv/bin/activate
run-task <nom de la pràctica>
```

### Opció 2: Executar el mòdul directament

```
source .venv/bin/activate
PYTHONPATH=src python -m <nom de la pràctica> 
```
