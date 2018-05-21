import glob
import numpy as np
import tensorflow as tf
from scipy.misc import imread
from abc import abstractmethod
from .utils import unpickle

class BaseDataset():
    def __init__(self, name, path, training=True, augment=True):
        self.name = name
        self.augment = augment and training
        self.training = training
        self.path = path
        self._data = []

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        total = len(self)
        start = 0

        while start < total:
            item = self[start]
            start += 1
            yield item

        raise StopIteration

    def __getitem__(self, index):
        val = self.data[index]
        img = imread(val) if isinstance(val, str) else val

        if self.augment and np.random.binomial(1, 0.5) == 1:
            img = img[:, ::-1, :]

        return img

    def generator(self, batch_size, recusrive=False):
        start = 0
        total = len(self)

        while True:
            while start < total:
                end = np.min([start + batch_size, total])
                items = np.array([self[item] for item in range(start, end)])
                start = end
                yield items

            if recusrive:
                start = 0

            else:
                raise StopIteration


    @property
    def data(self):
        if len(self._data) == 0:
            self._data = self.load()
            np.random.shuffle(self._data)

        return self._data

    @abstractmethod
    def load(self):
        return []


class Dataset(BaseDataset):
    def __init__(self, path, training=True, augment=True):
        super(Dataset, self).__init__(path, training, augment)

    def load(self):
        if self.training:
            data = np.array(
                glob.glob(self.path + '/train/*.jpg', recursive=True))

        else:
            data = np.array(glob.glob(self.path + '/test/*.jpg'))

        return data
