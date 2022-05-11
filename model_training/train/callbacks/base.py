import pytorch_lightning as pl
from abc import abstractmethod, ABC


class BaseCallback(pl.Callback, ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config):
        pass
