from framework.model import ImageEncoder, TextEncoder

class NaiveImageEncoder(ImageEncoder):
    def __init__(self, config):
        self.config = config
        self.model = None
    
    def load(self, model_path):
        return None
    
    def save(self, model_path):
        return None
    
    def encode(self, data):
        return data
    
class NaiveTextEncoder(TextEncoder):
    def __init__(self, config):
        self.config = config
        self.model = None
    
    def load(self, model_path):
        return None
    
    def save(self, model_path):
        return None
    
    def encode(self, data):
        return data