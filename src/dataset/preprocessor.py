from torchvision import transforms

class Preprocessor:
    def __init__(self, image_size):

        self.transform_list = []
        self.zoom_transform_list = []

        # self.transform_list.append(transforms.Gra)
        self.transform_list.append(transforms.Resize(image_size))

        self.zoom_transform_list.append(transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)))

        # self.transform_list.append(transforms.RandomCrop(image_size))

        self.transform_list.append(transforms.ColorJitter(contrast=(0.8, 1.5)))
        self.zoom_transform_list.append(transforms.ColorJitter(contrast=(0.8, 1.5)))

        self.transform_list.append(transforms.RandomHorizontalFlip(p=0.2))
        self.zoom_transform_list.append(transforms.RandomHorizontalFlip(p=0.2))

        self.transform_list.append(transforms.RandomRotation(5))
        self.zoom_transform_list.append(transforms.RandomRotation(5))

        self.transform_list.append(transforms.ToTensor())
        self.zoom_transform_list.append(transforms.ToTensor())



    def get_zoom_in_transform(self):
        return transforms.Compose(self.zoom_transform_list)


    def get_original_transforms(self):
        return transforms.Compose(self.transform_list)