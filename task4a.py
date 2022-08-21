import pathlib
import matplotlib.pyplot as plt
import utils
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer
import torchvision


class Model(nn.Module):
    def __init__(self, image_channels, num_classes):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10) # No need to apply softmax,
        # as this is done in nn.CrossEntropyLoss
        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected
            param.requires_grad = True # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
            param.requires_grad = True # layers
        
        num_filters = self.model.fc.in_features

        self.num_classes = num_classes
        #"""
        #last conv layers:
        self.model.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ),nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        #"""
        self.num_output_features = 512*1 #*4*4 # a little unsure why not to multiply x4x4 here since i though output size of image was 4x4, but i guess it is 1x1 (saw from error when program was calculating matrices from fc)
        #"""
        self.model.fc = nn.Sequential(
            nn.Linear(self.num_output_features, 64), 
            nn.ReLU(),
            nn.Linear(64, 10) 
        )
        #"""

    def forward(self, x):
        x = self.model(x)
        return x


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = Model(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    trainer.train()
    #create_plots(trainer, "task4a")
        

if __name__ == "__main__":
    main()