from streamlit import image

from model.image_model import SceneModel
from dataset.custom_dataset import SceneDataset
from dataset.preprocessor import Preprocessor
from torch.utils.data import DataLoader, ConcatDataset

import lightning as L

# ======================================

epoch = 45

image_size = (128  , 128)
lr = 0.004


# =======================================
preprocessor = Preprocessor(image_size)
transform = preprocessor.get_original_transforms()
zoom_transform = preprocessor.get_zoom_in_transform()

train_dataset = SceneDataset("../data", "seg_", "train", transform)
train_zoom_in_dataset= SceneDataset("../data", "seg_", "train", zoom_transform)

combined_test_dataset = ConcatDataset([train_dataset, train_zoom_in_dataset])

test_dataset = SceneDataset("../data", "seg_", "test", transform)


train_dataloader = DataLoader(combined_test_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

model = SceneModel(class_num=6, image_size=image_size, lr=lr)

trainer = L.Trainer(max_epochs=epoch)
trainer.fit(model, train_dataloader)
trainer.test(model, test_dataloader)




