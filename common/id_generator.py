import base64
from abc import ABCMeta, abstractmethod

from typing import Tuple


class IdGenerator(metaclass=ABCMeta):
    def __init__(self):
        self._counter = 0

    @abstractmethod
    def id(self) -> str:
        raise AttributeError('interface method')

    def gen_order_id(self) -> str:
        name = self.id() + ';' + str(self._counter)
        self._counter += 1
        return base64.b64encode(name.encode('ascii')).decode('ascii')

    @staticmethod
    def reveal_owner(order_id: str) -> Tuple:
        name, num = base64.b64decode(order_id.encode('ascii')).decode('ascii').split(';')  # type: str
        return name, int(num)
