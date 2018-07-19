from enum import Enum


class Symbol(Enum):
    XBTUSD = {'tick_size': 0.5}

    __iter__ = Enum.__iter__

    def __str__(self):
        return self.name

    @property
    def tick_size(self):
        return self.value['tick_size']

    @staticmethod
    def value_of(name):
        try:
            return Symbol[name]
        except KeyError:
            return None
