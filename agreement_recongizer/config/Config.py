import configparser


class Config:
    __config: configparser = None
    batch_size: str = None
    epoch: str = None
    learning_rate: str = None

    def __init__(self):
        self.__config = configparser.ConfigParser()
        self.__config.read('config.ini')
        self.batch_size = self.__config['train']['batch_size']
        self.epoch = self.__config['train']['epoch']
        self.lr = self.__config['train']['learning_rate']

