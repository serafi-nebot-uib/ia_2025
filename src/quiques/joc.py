import enum
from importlib.resources import files

import pygame

from iaLib import agent, joc



class Joc(joc.Joc):
    def __init__(self, agents: list[agent.Agent]):
        super(Joc, self).__init__(agents, (1024, 512), title="Casa")

        self.__illes = {
            "ESQ": {"LLOP": 3, "POLL": 3},
            "DRET": {"LLOP": 0, "POLL": 0},
        }

        self.__lloc = "ESQ"

    @staticmethod
    def altre_lloc(lloc: str):
        if lloc == "ESQ":
            return "DRET"
        else:
            return "ESQ"

    def _aplica(self, accio, params=None, agent_actual=None) -> None:
        if accio not in ("A", "M"):
            raise ValueError(f"Acció no existent en aquest joc: {accio}")

        if accio == "M":
            if params is None or len(params) != 2:
                raise ValueError(
                    "Paràmetres incorrectes: has d'indicar el nombre de llops i polls a moure"
                )

            moviment_polls, moviment_llop = params

            if moviment_llop + moviment_polls > 2:
                raise agent.Trampes()

            lloc_dif = Joc.altre_lloc(self.__lloc)

            self.__illes[self.__lloc]["LLOP"] -= moviment_llop
            self.__illes[lloc_dif]["LLOP"] += moviment_llop

            self.__illes[self.__lloc]["POLL"] -= moviment_polls
            self.__illes[lloc_dif]["POLL"] += moviment_polls

            self.__lloc = lloc_dif

            for illa in self.__illes.values():
                if (illa["LLOP"] > illa["POLL"]) and (illa["POLL"] != 0):
                    raise joc.HasPerdut("Els llops s'han menjat els polls")

    def _draw(self) -> None:
        super(Joc, self)._draw()
        window = self._game_window
        window.fill(pygame.Color(88, 189, 247))

        pygame.draw.rect(
            window, pygame.Color(5, 243, 255), pygame.Rect(0, 256, 1024, 256)
        )

        img_path = files("quiques.images") / "illa.png"
        img = pygame.image.load(img_path)
        img = pygame.transform.scale(img, (200, 200))
        window.blit(img, (20, 150))

        img_path = files("quiques.images") / "illa-r.png"
        img = pygame.image.load(img_path)
        img = pygame.transform.scale(img, (200, 200))
        window.blit(img, (824, 150))

        barca_path = files("quiques.images") / "barca.png"
        barca = pygame.image.load(barca_path)
        barca = pygame.transform.scale(barca, (100, 100))

        if self.__lloc == "ESQ":
            window.blit(barca, (240, 250))
        else:
            window.blit(barca, (824 - 100, 250))

        llop_path = files("quiques.images") / "llop.png"
        llop = pygame.image.load(llop_path)
        llop = pygame.transform.scale(llop, (100, 50))

        poll_path = files("quiques.images") / "gallina.png"
        poll = pygame.image.load(poll_path)
        poll = pygame.transform.scale(poll, (50, 50))

        for i, illa in enumerate(self.__illes.values()):
            for i_llop in range(illa["LLOP"]):
                if i == 0:
                    window.blit(llop, (20 + (i_llop * 25), 300))
                else:
                    window.blit(llop, (824 - (i_llop * 25), 300))
            for i_poll in range(illa["POLL"]):
                if i == 0:
                    window.blit(poll, (20 + (i_poll * 25), 300))
                else:
                    window.blit(poll, (824 - (i_poll * 25), 300))

    def percepcio(self) -> dict:
        return {
            "Lloc": self.__lloc,
            "Poll Esq": self.__illes["ESQ"]["POLL"],
            "Llop Esq": self.__illes["ESQ"]["LLOP"]
        }
