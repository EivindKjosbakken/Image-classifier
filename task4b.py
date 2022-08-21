import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
from torch import nn

import numpy as np
#image = Image.open("images/zebra.jpg")
image = Image.open(r"C:\Users\eivin\NTNU vår 22 lokal\DDL\TDT4265_StarterCode-main\assignment3\images\zebra.jpg")
plt.imsave("task4img.png", image)
print("Image shape:", image.size)



model = torchvision.models.resnet18(pretrained=True)
#print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    #må kanskje ha denne
    #image = image.squeeze()
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image


indices = [14, 26, 32, 49, 52]



#task 4b:
activation = torch.squeeze(activation) #need to remove the useless layer of batch size, so I can plot the image
for indice in indices:
    #plotting the filter weight:
    weightImg = first_conv_layer.weight[indice, :, :, :] #weight has shape: (num_filters, color_channels, H, W), I want the filter number of indice
    weightImg = torch_image_to_numpy(weightImg)
    plt.imsave(f"zebraWeightWithIndice{indice}.png", weightImg)

    #plotting the activation:
    actImg = activation[indice, :, :] #after squeeze, activation has shape (num_filters, H, W), we want filter number indice like for the weight above
    actImg = torch_image_to_numpy(actImg)
    plt.imsave(f"zebraActivationWithIndice{indice}.png", actImg) 
    

