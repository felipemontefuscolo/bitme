import base64
import uuid
from abc import ABCMeta, abstractmethod


class IdGenerator(metaclass=ABCMeta):
    def __init__(self, use_uuid=False):
        self._use_uuid = use_uuid
        if not use_uuid:
            self._counter = -1

    @abstractmethod
    def id(self) -> str:
        raise AttributeError('interface class')

    def generate_id(self) -> str:
        if self._use_uuid:
            return "{}_{}".format(self.id(), base64.b64encode(uuid.uuid4().bytes).decode('utf8').rstrip('=\n'))
        else:
            self._counter += 1
            return "{}_{}".format(self.id(), self._counter)
