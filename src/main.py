#
# from dataset.custom_dataset import SceneDataset
# from dataset.preprocessor import Preprocessor
# from torch.utils.data import DataLoader
# import lightning as l
#
#
# transform = Preprocessor().get_transforms()
# dataset = SceneDataset("../data", "seg_", "train", transform)
#
#
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.style.use('ggplot')

import torch
from fastai.vision import *
from fastai.metrics import *

np.random.seed(7)
torch.cuda.manual_seed_all(7)

import os
print(os.listdir("../input"))