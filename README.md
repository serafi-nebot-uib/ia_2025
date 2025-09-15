# Intel·ligència artificial - UIB (2025-2026)

En aquest repositori trobareu tots els exemples pràctics treballats durant les classes de l’assignatura d’Intel·ligència Artificial (21722).
## Instal·lació

Per poder emprar aquests exemples necessitem primerament tenir un entorn de ``Python`` amb les 
llibreries instal·lades. Per fer-ho s'han de seguir les següents passes. Emprarem ``uv``: 

Per poder dur a terme aquest tutorial s'ha de tenir instal·lat `git` al vostre dispositiu. Aquí teniu l'[enllaç](https://git-scm.com/)
per fer-ho.

### 1. Clonar el repositori 

Descarregar el repositori.
```
     git clone https://github.com/miquelmn/ia_2025
```

### 2. Instal·lar `uv`:

Instal·lar [``uv``](https://docs.astral.sh/uv/), el gestor de paquets que emprarem.


### 3. Crear i activar l'entorn

**IMPORTANT:** Les següents instruccións s'han de fer dins la carpeta del repositori, descarregada al primer pas.


Crear l'entorn (només s'ha de fer el primer pic):
```
uv venv .venv
```

Activar l'entorn (Linux i MacOS):
```
source .venv/bin/activate
```

Windows:
```
.venv\Scripts\activate
```

Instal·lar llibreries (només s'ha de fer el primer pic):

```
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

Activar l'entorn (Linux i MacOS):
```
source .venv/bin/activate
```

Windows:
```
.venv\Scripts\activate
```

Executa una tasca
```
run-task <nom de la pràctica>
```

### Opció 2: Executar el mòdul directament

Activar l'entorn (Linux i MacOS):
```
source .venv/bin/activate
```

Windows:
```
.venv\Scripts\activate
```

Executa una tasca
```
PYTHONPATH=src python -m <nom de la pràctica> 
```
