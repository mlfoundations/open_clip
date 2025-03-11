from framework.preprocessor import Preprocessor

class NaivePreprocessor1(Preprocessor):
    def __init__(self, config):
        self.config = config
    
    def preprocess(self, dataset):
        img_data = dataset.img_data
        text_data = dataset.text_data
        
        # -------------
        # some preprocessing code here
        # -------------
        
        dataset.img_data = img_data
        dataset.text_data = text_data
        
        return dataset

class NaivePreprocessor2(Preprocessor):
    def __init__(self, config):
        self.config = config
    
    def preprocess(self, dataset):
        img_data = dataset.img_data
        text_data = dataset.text_data
        
        # -------------
        # some preprocessing code here
        # -------------
        
        dataset.img_data = img_data
        dataset.text_data = text_data

        return dataset