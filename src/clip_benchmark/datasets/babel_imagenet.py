import torchvision

"""
BabelImageNet from https://arxiv.org/pdf/2306.08658.pdf
Adapted from https://github.com/gregor-ge/Babel-ImageNet, thanks to the authors
"""


class BabelImageNet(torchvision.datasets.ImageNet):
    def __init__(
        self, root: str, idxs, split: str = 'val', download=None, **kwargs
    ) -> None:
        super().__init__(root, split, **kwargs)
        examples_per_class = len(self.targets) // 1000
        select_idxs = [
            idx * examples_per_class + i
            for idx in idxs
            for i in range(examples_per_class)
        ]
        self.targets = [i for i in range(len(idxs)) for _ in range(examples_per_class)]
        self.imgs = [self.imgs[i] for i in select_idxs]
        self.samples = [self.samples[i] for i in select_idxs]
        self.idxs = idxs

    def __getitem__(self, i):
        img, target = super().__getitem__(i)
        target = self.idxs.index(target)
        return img, target
