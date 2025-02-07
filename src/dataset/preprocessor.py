from torchvision import transforms

class Preprocessor:
    def __init__(self):

        self.transform_list = []

        self.transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        self.transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        self.transform_list.append(transforms.RandomRotation(5))

        self.transform_list.append(transforms.ToTensor())
        self.transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))


    def get_transforms(self):
        return transforms.Compose(self.transform_list)