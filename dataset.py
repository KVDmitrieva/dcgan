import os

from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms


class AnimeDataset(Dataset):
    def __init__(self, data_path, train=True, train_ratio=0.8, limit=None):
        super().__init__()

        self.data_path = data_path
        self.index = os.listdir(data_path)

        if limit is not None:
            self.index = self.index[:limit]

        sep_ind = int(len(self.index) * train_ratio)
        self.index = self.index[:sep_ind] if train else self.index[sep_ind:]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        path = self.data_path + '/' + self.index[idx]
        real_image = read_image(path)
        image_resized = self.transform(real_image) * 2 - 1
        return image_resized
