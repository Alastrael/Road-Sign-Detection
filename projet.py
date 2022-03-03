import numpy as np
import torch as torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# TO PLOT TENSORS AS IMAGES
#print(CLASSES.get(int(data.classes[0].title())), plt.imshow(data[0][0].permute(1,2,0)))

DATA_FOLDER = "panneaux_route"
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

######### IMPORT DATA #########
def import_train_data() :
    trans = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    train_data = datasets.ImageFolder("panneaux_route/Train", transform=trans)
    return train_data

def import_test_data() :
    df = pd.read_csv('panneaux_route/Test.csv', sep=',')
    df_iter = df[['ClassId', 'Path']]
    return df_iter

train_data = import_train_data()
test_data = import_test_data()

######### CONVERT TO TENSORS #########
def train_convert_to_tensors() :
    data = []
    labels = []
    for a, b in train_data :
        data.append(a)
        labels.append(b)
    inputs = data                   # tableau de tenseurs
    targets = torch.tensor(labels)  # tenseurs des etiquettes des inputs
    return inputs, targets

def test_convert_to_tensors(df_iter) :
    trans = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
    inputs = []
    targets = []
    for index, row in df_iter.iterrows() :
        img = Image.open('panneaux_route/'+row['Path'])
        inputs.append(trans(img))
        targets.append(row['ClassId'])
    return inputs, targets

######### DATALOADER CREATION #########
from torch.utils.data import Dataset, DataLoader
class MyDataSet(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

train_inputs, train_targets = train_convert_to_tensors()
test_inputs, test_targets = test_convert_to_tensors(test_data)
#print(train_inputs[0], train_targets[0])
train_loader = DataLoader(MyDataSet(train_inputs, train_targets), batch_size=256)
validation_loader = DataLoader(MyDataSet(test_inputs, test_targets), batch_size=256)

######### MODEL(S) #########
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 43)

    def forward(self, input):
        layer1 = F.relu(self.conv1(input))                          
        layer2 = F.max_pool2d(layer1, kernel_size=(2, 2), stride=2) 
        layer3 = F.relu(self.conv2(layer2))                         
        layer4 = F.max_pool2d(layer3, kernel_size=(2, 2), stride=2) 
        layer5 = F.relu(self.conv3(layer4))                         
        layer6 = F.relu(self.fc1(torch.flatten(layer5,1)))          
        layer7 = self.fc2(layer6)                                   
        return layer7

lenet = LeNet5()
print("lenet parameters:")
num_param = 0
for p in lenet.parameters():
    print(p.size())
    num_param += torch.tensor(p.size()).prod().item()
print("total lenet parameters:", num_param)

def train_loop(train_loader, model, loss_map, lr=1e-3, epochs=20):
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Train model
    for epoch in range(epochs):
        loss_epoch = 0.
        for images, labels in train_loader:
            # Transfers data to GPU
            images, labels = images.to(device), labels.to(device)
            # Primal computation
            output = model(images)            
            loss = loss_map(output, labels)            
            # Gradient computation
            model.zero_grad()
            loss.backward()
            # perform parameter update based on current gradients
            optimizer.step()
            # compute the epoch training loss
            loss_epoch += loss.item()
        # display the epoch training loss
        print(f"epoch : {epoch + 1}/{epochs}, loss = {loss_epoch:.6f}")
            

def validate(data_loader, model):
    nb_errors = 0
    nb_tests = 0
    device = next(model.parameters()).device # current model device
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device) # move data same model device
        output = model(images)
        nb_errors += ((output.argmax(1)) != labels).sum()
        nb_tests += len(images)
    
    return (100*(nb_tests-nb_errors)) // nb_tests

print(f"Lenet before learning, accuracy = {validate(validation_loader, lenet)}%")
train_loop(
    train_loader=train_loader, 
    model=lenet, 
    loss_map=nn.CrossEntropyLoss(),
    lr=1e-3,
    epochs=20)
print(f"Lenet after learning, accuracy = {validate(validation_loader, lenet)}%")