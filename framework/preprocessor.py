

class Preprocessor:
    def __init__(self, config):
        # config is a dict containing the configuration for the preprocessor
        # should be defined in the config yaml file
        self.config = config

    def preprocess(self, dataset):
        # Should return preprocessed dataset
        raise NotImplementedError
