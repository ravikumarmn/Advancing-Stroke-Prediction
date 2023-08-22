import os
from PIL import Image
from torch.utils.data import Dataset

class CTMRIClassificationDataset(Dataset):
    def __init__(self, data_root, transform=None, train=True):
        self.data_root = data_root
        self.transform = transform
        self.class_mapping = {'trainA': 1, 'trainB': 0, 'testA': 1, 'testB': 0}
        self.train = train
        self.data = self.load_data()

    def load_data(self):
        data = []
        if self.train:
            class_names = ['trainA', 'trainB']
        else:
            class_names = ['testA', 'testB']

        for class_name in class_names:
            class_path = os.path.join(self.data_root, class_name)
            class_label = self.class_mapping[class_name]
            file_names = os.listdir(class_path)
            for file_name in file_names:
                file_path = os.path.join(class_path, file_name)
                data.append((file_path, class_label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
