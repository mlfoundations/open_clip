from framework.dataset import Dataset

class NaiveDataset(Dataset):
    def __init__(self, config):
        self.config = config
        
        self.img_data = ['img1', 'img2', 'img3']
        self.text_data = ['text1', 'text2', 'text3']
    
    def __len__(self):
        return len(self.img_data)
    
    def __getitem__(self, idx):
        return self.img_data[idx], self.text_data[idx]