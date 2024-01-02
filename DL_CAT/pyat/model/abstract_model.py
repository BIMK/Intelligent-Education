from abc import ABC, abstractmethod
from ..utils.data import TrainDataset, AdapTestDataset, _Dataset


class AbstractModel(ABC):

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def adaptest_update(self, adaptest_data: AdapTestDataset):
        raise NotImplementedError

    @abstractmethod
    def adaptest_evaluate(self, adaptest_data: AdapTestDataset):
        raise NotImplementedError

    @abstractmethod
    def adaptest_init(self, data: _Dataset):
        raise NotImplementedError

    @abstractmethod
    def adaptest_train(self, train_data: TrainDataset):
        raise NotImplementedError

    @abstractmethod
    def adaptest_save(self, path):
        raise NotImplementedError

    @abstractmethod
    def adaptest_preload(self, path):
        raise NotImplementedError