from torch.utils.data import Dataset as TorchDataset

class Dataset(TorchDataset):
    def __init__(self, config):
        self.config = config
        
        # Dataset class should have the following attributes:
        # - img_data: list of image data samples
        # - text_data: list of text data samples
        self.img_data = None 
        self.text_data = None

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Args:
            idx: Index of the sample
        Returns:
            Tuple: ()
        """
        raise NotImplementedError