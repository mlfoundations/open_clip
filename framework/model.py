
class Model:
    def __init__(self, config):
        # config is a dict containing the configuration for the model
        # Model Object should have the following attributes:
        # - config: configuration for the model
        # - model: the callable model object
        self.config = config
        self.model = None
    
    def load(self, model_path):
        raise NotImplementedError("load your model to self.model")
    
    def save(self, model_path):
        raise NotImplementedError("save self.model to model_path")
    
    def encode(self, data):
        raise NotImplementedError
    
    def to(self, device):
        self.model.to(device)
     

class ImageEncoder(Model):
    def __init__(self, config):
        self.config = config
        self.model = None

    
class TextEncoder(Model):
    def __init__(self, config):
        self.config = config
        self.model = None
