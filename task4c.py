import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
from torch import nn

import numpy as np
#image = Image.open("images/zebra.jpg")
image = Image.open(r"C:\Users\eivin\NTNU vår 22 lokal\DDL\TDT4265_StarterCode-main\assignment3\images\zebra.jpg")

print("Image shape:", image.size)



model = torchvision.models.resnet18(pretrained=True)

layer1 = model.layer1
layer2 = model.layer2
layer3 = model.layer3

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]


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



#task 4c:
children = nn.Sequential(*list(model.children())) #basically making a new model of all layers
myModel = children[:-2] #then taking all the layers, but not the last 2
act = myModel(image) #then fowarding through my model I just made in the 2 lines above
print(act.size()) #can see the shape is (1, 512, 7, 7) as it should be
act = torch.squeeze(act) #removing the useless 1 in the shape of act
for i in range(10):
    actImg = act[i,:,:] #want filter number i, and visualize the filter from there
    actImg = torch_image_to_numpy(actImg)
    plt.imsave(f"task4cFilter{i}.png", actImg)



