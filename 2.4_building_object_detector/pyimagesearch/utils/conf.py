# import commentjson as json
import json

class Conf:
    def __init__(self, confPath):
        conf = json.loads(open(confPath).read())
        # update add conf to existing dict
        self.__dict__.update(conf)

    def __getitem__(self, k):
        # implement slicing method
        return self.__dict__.get(k, None)