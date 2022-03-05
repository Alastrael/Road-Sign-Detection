import numpy as np
import torch as torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.nn.functional as F
from torch.utils import data as data

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

######### DATA SETS #########
from torch.utils.data import Dataset, DataLoader

trainingset = datasets.ImageFolder("panneaux_route/Train", transform=transforms.Compose(
    [transforms.Resize((32,32)),
    transforms.ToTensor()]
))

print(trainingset.classes)
print(CLASSES.get(int(trainingset.classes[trainingset[5005][1]]))), plt.imshow(trainingset[5005][0].permute(1,2,0))

trainSize = int(0.8 * len(trainingset))
validationSize = len(trainingset) - trainSize
trainDataset, validationDataset = data.random_split(trainingset, [trainSize, validationSize])

trainLoader = DataLoader(trainDataset, batch_size=256, shuffle=True)
validationLoader = DataLoader(validationDataset, batch_size=1024)

######### COMBIEN DE DONNEES PAR CLASSE #########
def check_repartition(dico, dataset) :
    for images, label in dataset :
        if label in dico :
            dico[label] += 1
        else :
            dico[label] = 1

######### VERIFIONS LA REPARTITION DES DONNEES #########
check_init = {}
check_repartition(check_init, trainingset)
print(check_init)

check_traindataset = {}
check_repartition(check_traindataset, trainDataset)
print(check_traindataset)

X = np.arange(len(check_init))
ax = plt.subplot(111)
ax.bar(X, check_init.values(), width=0.4, color='b', align='center')
ax.bar(X-0.5, check_traindataset.values(), width=0.4, color='g', align='center')
ax.legend(('ImageFolder','TrainDataset'))
plt.xticks(X, check_init.keys())
plt.title("Repartition", fontsize=17)

plt.show()

def import_test_data() :
    df = pd.read_csv('panneaux_route/Test.csv', sep=',')
    df_iter = df[['ClassId', 'Path']]
    
    trans = transforms.Compose([transforms.Resize((34,34)), transforms.ToTensor()])
    data = []
    for index, row in df_iter.iterrows() :
        img = Image.open('panneaux_route/'+row['Path'])
        data.append((trans(img), row['ClassId']))
    return data
test_data = import_test_data()
print(type(test_data))

######### TRAIN LOOP #########
def train_loop(train_loader, model, loss_map, lr=1e-3, epochs=20):
    history = []
    # use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    # create optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    # Train model
    for epoch in range(epochs):
        loss_epoch = 0.
        for images, labels in train_loader:
            # Transfers data to GPU
            #images, labels = images.to(device), labels.to(device)
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
        test_acc = validate(validationLoader, model)
        history.append(
            {'epoch' : epoch + 1,
             'loss' : loss_epoch,
             'test_acc' : test_acc})
        print(f"epoch : {epoch + 1}/{epochs}, loss = {loss_epoch:.6f}, test_acc = {test_acc}%")
    return history

######### VALIDATION #########
def validate(data_loader, model):
    nb_errors = 0
    nb_tests = 0
    device = next(model.parameters()).device # current model device
    for i, (images, labels) in enumerate(data_loader):
        #images, labels = images.to(device), labels.to(device) # move data same model device
        output = model(images)
        nb_errors += ((output.argmax(1)) != labels).sum()
        nb_tests += len(images)
    
    return (100*(nb_tests-nb_errors)) / nb_tests

######### MODEL(S) #########
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5))
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

print(f"Lenet before learning, accuracy = {validate(validationLoader, lenet)}%")
h = train_loop(
    train_loader=trainLoader, 
    model=lenet, 
    loss_map=nn.CrossEntropyLoss(),
    lr=0.001,
    epochs=15)
#print(h)
print(f"Lenet after learning, accuracy = {validate(validationLoader, lenet)}%")

def show_learning(history):
    fig, ax = plt.subplots()
    ax.plot([h['epoch'] for h in history], [h['loss'] for h in history], label='loss')
    ax.plot([h['epoch'] for h in history], [h['test_acc'] for h in history], label='test accuracy')
    plt.legend()
    plt.show()
show_learning(h)