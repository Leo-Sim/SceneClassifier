from torch import optim, nn

import torch
import lightning as L
import matplotlib.pyplot as plt

from torchvision.models import resnet50

from torchmetrics import Accuracy, Precision, Recall, F1Score



class SceneModel(L.LightningModule):

    def __init__(self, class_num, image_size, lr=0.003, momentum=0.9, weight_decay=0.01):
        super().__init__()

        self.loss_list = []
        self.accuracy_list = []

        # self.resnet = resnet50(pretrained=False)

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        conv_output_size = self._get_conv_output_size((128,128))
        # conv_output_size = self._get_conv_output_size(image_size)

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4096, bias=True),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, class_num, bias=True),
        )


        # Set default optimizer and loss function
        # self.optimizer = optim.Adam(self.resnet.parameters(), lr=lr, weight_decay=weight_decay)
        self.optimizer = optim.Adam(self.fc_layer.parameters(), lr=lr)
        self.loss_function = nn.CrossEntropyLoss()

        # this is to get the test results
        self.f1_value = F1Score(task="multiclass", num_classes=class_num, average='macro')
        self.precision_value = Precision(task="multiclass", num_classes=class_num,
                                         average='macro')
        self.recall_value = Recall(task="multiclass", num_classes=class_num, average='macro')
        self.accuracy_value = Accuracy(task="multiclass", num_classes=class_num)

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
        return self.optimizer

    def forward(self, x):
        # return self.resnet(x)
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

        loss = self.loss_function(y_hat, y)
        f1 = self.f1_value(prediction, y)
        precision = self.precision_value(prediction, y)
        accuracy = self.accuracy_value(prediction, y)
        recall = self.recall_value(prediction, y)

        # save data for tensorboard
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
        print("loss : ",loss)
        self.loss_list.append(loss)

    def on_test_epoch_end(self):
        # 손실 그래프
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_list, label="Loss", marker='o', linestyle='-')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss per Epoch")
        plt.legend()
        plt.grid()
        plt.show()


