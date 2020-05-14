import configparser
import os


class Config:
    __config: configparser = None
    batch_size: str = None
    epoch: str = None
    learning_rate: str = None

    def __init__(self):
        self.__config = configparser.ConfigParser()
        path = '/'.join((os.path.abspath(__file__).replace('\\', '/')).split('/')[:-1])
        self.__config.read(os.path.join(path, 'config.ini'))
        self.batch_size = self.__config.get('train', 'batch_size')
        self.epoch = self.__config['train']['epoch']
        self.lr = self.__config['train']['learning_rate']
