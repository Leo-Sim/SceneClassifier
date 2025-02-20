from torch import optim, nn

import torch
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns


from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchmetrics import ConfusionMatrix
from torch.optim.lr_scheduler import StepLR



class SceneModel(L.LightningModule):

    def __init__(self, class_num, image_size, lr=0.003, momentum=0.9, weight_decay=0.01, label_detail={}):
        super().__init__()

        self.loss_list = []
        self.accuracy_list = []

        # self.class_names = class_names if class_names else [str(i) for i in range(class_num)]
        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=class_num)
        self.label_detail = label_detail

        # if label_detail is None:
        self.index_to_class = {v: k for k, v in self.label_detail.items()}


        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        conv_output_size = self._get_conv_output_size(image_size)

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, class_num, bias=True),
        )

        self.weight_decay = weight_decay
        self.lr = lr

        self.loss_function = nn.CrossEntropyLoss()

        # this is to get the test results
        self.f1_value = F1Score(task="multiclass", num_classes=class_num, average='macro')
        self.precision_value = Precision(task="multiclass", num_classes=class_num,
                                         average='macro')
        self.recall_value = Recall(task="multiclass", num_classes=class_num, average='macro')
        self.accuracy_value = Accuracy(task="multiclass", num_classes=class_num)

    def _get_class_text(self, preds):
        return [self.index_to_class[p] for p in preds.tolist()]

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def _get_conv_output_size(self, input_size):
        """
        Automatically calculate the output size from convolution layer
        :param input_size:
        :return: int
        """

        x = torch.randn(1, 3, *input_size)
        x = self.conv_layer(x)
        output_size = x.size(1) * x.size(2) * x.size(3)
        return output_size

    def configure_optimizers(self):

        optimizer = optim.Adam(self.fc_layer.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }

    def forward(self, x):

        x = self.conv_layer(x)
        output = self.fc_layer(x)

        # FC layer


        return output

    def training_step(self, batch, batch_idx):
        x, label = batch
        output = self(x)

        loss = self.loss_function(output, label)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        prediction = torch.argmax(y_hat, dim=1)

        self.confusion_matrix.update(prediction, y)

        loss = self.loss_function(y_hat, y)
        f1 = self.f1_value(prediction, y)
        precision = self.precision_value(prediction, y)
        accuracy = self.accuracy_value(prediction, y)
        recall = self.recall_value(prediction, y)

        self.log("loss", loss)
        self.log("f1_value", f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("precision_value", precision, prog_bar=True, on_step=False, on_epoch=True)
        self.log("accuracy_value", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("recall_value", recall, prog_bar=True, on_step=False, on_epoch=True)
        # return matrix value for further analysis
        return {'test_loss': loss, 'test_f1': f1, 'test_precision': precision, 'test_accuracy': accuracy,
                'test_recall': recall}

    def on_train_epoch_end(self):

        loss = self.trainer.callback_metrics['train_loss'].item()

        self.loss_list.append(loss)

    def on_test_epoch_end(self):

        cm = self.confusion_matrix.compute().cpu().numpy()

        fig = self._plot_confusion_matrix(cm, self.index_to_class.values())

        self.confusion_matrix.reset()

        # 손실 그래프
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_list, label="Loss", marker='o', linestyle='-')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss per Epoch")
        plt.legend()
        plt.grid()
        plt.show()

    def _plot_confusion_matrix(self, cm, class_names):

        fig, ax = plt.subplots(figsize=(6, 6))

        class_labels = [self.index_to_class[i] for i in range(len(self.index_to_class))]

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
        return fig
