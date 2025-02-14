
import PIL.Image as Image
import os
import numpy as np

from torch.utils.data import Dataset



class SceneDataset(Dataset):
    def __init__(self, path, dir_prefix, goal, transform):

        self.goal = goal
        self.dir_prefix = dir_prefix
        self.path = path

        self.transform = transform

        self.label_info = {
            "buildings" : 0,
            "forest" : 1,
            "glacier" : 2,
            "mountain" : 3,
            "sea": 4,
            "street" : 5
        }


        self.data_list = []

        self._get_all_image_data()

    def get_label_detail(self):
        return self.label_info

    def _get_all_image_data(self):

        dir_name =  self.dir_prefix + self.goal.lower()
        data_path = os.path.join(self.path, dir_name, dir_name)

        dirs = os.listdir(data_path)

        for dir in dirs:
            sub_dir_path = os.path.join(data_path, dir)
            sub_dirs = os.listdir(sub_dir_path)

            for sub_dir in sub_dirs:
                data_info = self.ImageDetail(os.path.join(sub_dir_path,sub_dir), self.label_info[dir])
                self.data_list.append(data_info)


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, index):

        data_info = self.data_list[index]

        image = data_info.image
        label = data_info.label

        image_tensor = self.transform(image)

        return image_tensor, label

    class ImageDetail:
        def __init__(self, image_path, label):
            self._image_path = image_path
            self._label = label

        @property
        def image(self):
            return Image.open(self._image_path)

        @property
        def label(self):
            return self._label

        @label.setter
        def label(self, label):
            self._label = label


