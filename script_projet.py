from importlib.resources import path
import sys
import os

# LIBRAIRIES DE TORCH
import torch as torch
from torch.utils import data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam, SGD
import torch.nn.functional as F

# LIBRAIRIES DE TRAITEMENT DE DONNEES
import numpy as np
import pandas as pd
from functools import reduce 

# LIBRAIRIE DE TRAITEMENT D'IMAGES
from PIL import Image

# LIBRAIRIE DE PLOT
import matplotlib.pyplot as plt
import plotly.graph_objects as go
CLASSES = { 
    0:"Limitation de vitesse (20km/h)",
    1:"Limitation de vitesse (30km/h)", 
    2:"Limitation de vitesse (50km/h)", 
    3:"Limitation de vitesse (60km/h)", 
    4:"Limitation de vitesse (70km/h)", 
    5:"Limitation de vitesse (80km/h)", 
    6:"Fin de limitation de vitesse (80km/h)", 
    7:"Limitation de vitesse (100km/h)", 
    8:"Limitation de vitesse (120km/h)", 
    9:"Interdiction de depasser", 
    10:"Interdiction de depasser pour vehicules > 3.5t", 
    11:"Intersection ou' vous etes prioritaire", 
    12:"Route prioritaire", 
    13:"Ceder le passage", 
    14:"Arret a' l'intersection", 
    15:"Circulation interdite", 
    16:"Acces interdit aux vehicules > 3.5t", 
    17:"Sens interdit", 
    18:"Danger", 
    19:"Virage a' gauche", 
    20:"Virage a' droite", 
    21:"Succession de virages", 
    22:"Cassis ou dos-d'ane", 
    23:"Chaussee glissante", 
    24:"Chaussee retrecie par la droite", 
    25:"Travaux en cours", 
    26:"Annonce feux", 
    27:"Passage pietons", 
    28:"Endroit frequente' par les enfants", 
    29:"Debouche' de cyclistes", 
    30:"Neige ou glace",
    31:"Passage d'animaux sauvages", 
    32:"Fin des interdictions precedemment signalees", 
    33:"Direction obligatoire a' la prochaine intersection : a' droite", 
    34:"Direction obligatoire a' la prochaine intersection : a' gauche", 
    35:"Direction obligatoire a' la prochaine intersection : tout droit", 
    36:"Direction obligatoire a' la prochaine intersection : tout droit ou a' droite", 
    37:"Direction obligatoire a' la prochaine intersection : tout droit ou a' gauche", 
    38:"Contournement obligatoire de l'obstacle par la droite", 
    39:"Contournement obligatoire de l'obstacle par la gauche", 
    40:"Carrefour giratoire", 
    41:"Fin d'interdiction de depasser", 
    42:"Fin d'interdiction de depasser pour vehicules > 3.5t" 
}

class LeNet5(nn.Module):
    def __init__(self, dropout=0.0, input_canals=3):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(input_canals, 32, kernel_size=(5, 5))        # 3 * 9 * 5 * 5 + 9
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=(3, 3))      # 9 * 9 * 3 * 3 + 9
        self.dropout1 = nn.Dropout2d(p=dropout)
        self.batchnorm1 = nn.BatchNorm2d(32)                     # as attribute, for affine=True

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4,4))        # 9 * 32 * 3 * 3 + 32
        self.conv2_1 = nn.Conv2d(64, 64, kernel_size=(3, 3))    # 32 * 32 * 3 * 3 + 32
        self.dropout2 = nn.Dropout2d(p=dropout)
        self.batchnorm2 = nn.BatchNorm2d(64)                    # as attribute, for affine=True

        self.conv3 = nn.Conv2d(64, 92, kernel_size=(3,3))       # 32 * 64 * 3 * 3 + 92
        self.conv3_1 = nn.Conv2d(92, 92, kernel_size=(2, 2))    # 64 * 64 * 3 * 23 + 92
        self.dropout3 = nn.Dropout(p=dropout)
        self.batchnorm3 = nn.BatchNorm2d(92)                    # as attribute, for affine=True

        self.fc1 = nn.Linear(92, 98)                          # 120 * 98 + 98
        self.batchnorm4 = nn.BatchNorm1d(98)

        self.fc2 = nn.Linear(98, 43)                            # 98 * 43 + 43

    def forward(self, input):                                               # B * 3 * 32 * 32  
        layer1 = F.relu(self.conv1(input))                                  # B * 9 * 28 * 28    28 car T-K+1 : (32 - 5 + 1) * (32 - 5 + 1)
        layer1_2 = F.relu(self.conv1_2(layer1))                             # B * 9 * 26 * 26
        layer2 = F.max_pool2d(layer1_2, kernel_size=(2, 2), stride=2)       # B * 9 * 13 * 13
        layer2_d = self.dropout1(layer2)
        layer2_bn = self.batchnorm1(layer2_d)

        layer3 = F.relu(self.conv2(layer2_bn))                              # B * 32 * 10 * 10
        layer3_1 = F.relu(self.conv2_1(layer3))                             # B * 16 * 8 * 8
        layer4 = F.max_pool2d(layer3_1, kernel_size=(2, 2), stride=2)       # B * 16 * 4 * 4
        layer4_d = self.dropout1(layer4)
        layer4_bn = self.batchnorm2(layer4_d)

        layer5 = F.relu(self.conv3(layer4_bn))                              # B * 120 * 2 * 2
        layer5_1 = F.relu(self.conv3_1(layer5))
        layer5_d = self.dropout3(layer5_1)
        layer5_bn = self.batchnorm3(layer5_d)     

        layer6 = F.relu(self.fc1(torch.flatten(layer5_bn,1)))               # B * 92
        layer7_bn = self.batchnorm4(layer6)       

        layer8 = self.fc2(layer7_bn)                                        # B * 43
        return layer8

def __main__(args) :
    path_to_image = args[0]
    transformation = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])
    transformation_gr = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), transforms.Resize((32,32)), transforms.ToTensor()
    ])
    try:
        classes = pd.read_csv('classes.csv')
        classes = classes['0'].tolist()
        image = Image.open(path_to_image)
    except:
        print("An error as occured while opening the image : ", path_to_image)
    
    model_rgb = LeNet5(0.4)
    model_gr = LeNet5(0.4, 1)

    model_rgb.load_state_dict(torch.load("model_rgb.pt"), strict=False)
    model_gr.load_state_dict(torch.load("model_gr.pt"), strict=False)

    model_rgb.eval()
    model_gr.eval()

    img_rgb = transformation(image)
    img_gr = transformation_gr(image)

    prediction_rgb = model_rgb(img_rgb.view(-1, 3, 32, 32)).argmax(1)
    prediction_gr = model_gr(img_gr.view(-1, 1, 32, 32)).argmax(1)

    plt.imshow(img_rgb.permute(1,2,0))
    
    print("Prediction du model en rgb :",CLASSES[int(classes[int(prediction_rgb)])], "(",int(classes[int(prediction_rgb)]),")", "\nPrediction du model en gris:", CLASSES[int(classes[int(prediction_gr)])], "(",int(classes[int(prediction_rgb)]),")")
    plt.show()
    

if __name__ == "__main__":
    __main__(sys.argv[1:])