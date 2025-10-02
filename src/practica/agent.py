from practica import joc


class Viatger(joc.Viatger):
    def __init__(self, *args, **kwargs):
        super(Viatger, self).__init__(*args, **kwargs)


    def actua(self, percepcio: dict):
        return "ESPERAR"
