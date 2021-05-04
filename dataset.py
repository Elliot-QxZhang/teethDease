import torchvision
from PIL import Image
from torch.utils.data import Dataset


class K_folder_dataset(Dataset):
    def __init__(self, data_list):
        self.imgs = data_list
        trans = [
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.50435138, 0.50435138, 0.50435138),
                                             (0.19735201, 0.19735201, 0.19735199))]
        self.transform = torchvision.transforms.Compose(trans)

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)