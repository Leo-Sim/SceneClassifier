from streamlit import image

from model.image_model import SceneModel
from dataset.custom_dataset import SceneDataset
from dataset.preprocessor import Preprocessor
from torch.utils.data import DataLoader, ConcatDataset

import numpy as np
import lightning as L
import matplotlib.pyplot as plt
import torchvision
import torch

# ======================================

def imshow(img,  title="", labels=None, predictions=None, label_detail=None):
    """이미지 출력 및 실제 레이블과 예측 레이블 표시"""
    npimg = img.numpy()
    plt.figure(figsize=(12, 12))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.title(title)

    if predictions is not None and label_detail is not None:
        # 레이블 추가
        for i in range(len(labels)):
            true_label = label_detail[int(labels[i].item())]
            pred_label = label_detail[int(predictions[i].item())]

            color = "green" if true_label == pred_label else "red"

            plt.text(
                10 + (i % 4) * 130,  # x 좌표
                10 + (i // 4) * 130,  # y 좌표
                f"T: {true_label}\nP: {pred_label}",
                fontsize=10,
                color=color,
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="black", boxstyle="round,pad=0.3"),
            )

    plt.show()


epoch = 50


image_size = (128  , 128)
lr = 0.003
weight_decay = 1e-5


# =======================================
preprocessor = Preprocessor(image_size)
transform = preprocessor.transforms()

train_dataset = SceneDataset("../data", "seg_", "train", transform)
test_dataset = SceneDataset("../data", "seg_", "test", transform)

# Print dataset size
print("================ Dataset Size ================")
print()
print("Train dataset : ", train_dataset.get_dataset_size())
print("Test dataset : ", test_dataset.get_dataset_size())
print()
print("==============================================")


train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# show sample training images
sample_data_iter = iter(train_dataloader)
images, labels = next(sample_data_iter)

label_detail = train_dataset.get_label_detail()

imshow(torchvision.utils.make_grid(images), "Sample images of test data")


model = SceneModel(class_num=6, image_size=image_size, lr=lr,weight_decay=weight_decay , label_detail=label_detail)

trainer = L.Trainer(max_epochs=epoch)
trainer.fit(model, train_dataloader)
trainer.test(model, test_dataloader)

sample_data_iter = iter(test_dataloader)
test_images, test_labels = next(sample_data_iter)


model.eval()
with torch.no_grad():
    test_outputs = model(test_images)
    _, predicted_labels = torch.max(test_outputs, 1)


imshow(torchvision.utils.make_grid(test_images, nrow=4), "Test Images with Predictions",
       test_labels, predicted_labels, test_dataset.get_label_num_as_index())






