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

if __name__ == '__main__':
    print("### IMPORTATION RUNNING ###")
else:
    # on commence par recolter les donnees et transformer les images en tensors
    raw_data = datasets.ImageFolder("panneaux_route/Train", transform=transforms.Compose(
        [transforms.Resize((32,32)),
        transforms.ToTensor()]
    ))

    # construisons le dataloader
    train_loader = DataLoader(raw_data, batch_size=1024, shuffle=True)

    df = pd.read_csv('panneaux_route/Test.csv', sep=',')
    transformation = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])

    labels = list(df['ClassId'])
    images = list(df['Path'])
    img_lst = []

    for img in images :
        img = "panneaux_route/" + img
        img_lst.append(transformation(Image.open(img)))

    class MyDataSet(Dataset):
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels
        def __len__(self):
            return len(self.inputs)
        def __getitem__(self, index):
            return self.inputs[index], self.labels[index]

    # le dataset
    test_dataset = MyDataSet(img_lst, labels)

    # le dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=256)

    print(raw_data.classes)

    for images, labels in train_loader :
        for i in range(5) :
            plt.imshow(images[i].permute(1,2,0))
            plt.show()
            print(CLASSES[int(raw_data.classes[labels[i]])]) # pour palier au probleme d'ordre des classes voici ce que nous devons faire
        break

    def repartition(dico, d, total) :
        for image, label in d :
            if raw_data.classes[label] in dico :
                dico[raw_data.classes[label]] += 1
            else :
                dico[raw_data.classes[label]] = 1
            total += 1

        for key in dico :
            dico[key] = 100*(dico[key] / total)

        my_list = dico.items()
        my_list = sorted(my_list)
        x, y = zip(*my_list)

        import plotly.graph_objects as go
        fig = go.Figure(
            data=[go.Bar(x=x, y=y)],
            layout_title_text="Nombre d'images par classe (en %)",
            layout_width=1000,
            layout_height=500
        )
        fig.show()
        return x, y

    raw_count = {}
    raw_total = 0

    x_r, y_r = repartition(raw_count, raw_data, raw_total)

    test_count = {}
    test_total = 0

    x_s, y_s = repartition(test_count, test_dataset, test_total)
    print(test_dataset.__len__())

    fig = go.Figure(
        data=[
            go.Bar(name='Train / Raw', x=x_r, y=y_r),
            go.Bar(name='Test', x=x_s, y=y_s)
        ],
        layout_title_text="Nombre d'images par classe (en %)",
        layout_width=1200,
        layout_height=500
    )
    fig.show()

    def validate_normal(predictions, labels) :
        nb_errors = ((predictions.argmax(1)) != labels).sum()
        return (len(predictions)-nb_errors).item()

    def validate_training(data_loader, model) :
        nb_errors = 0
        nb_tests = 0
        for i, (images, labels) in enumerate(data_loader):
            output = model(images)
            nb_errors += ((output.argmax(1)) != labels).sum()
            nb_tests += len(images)
        return (100*(nb_tests-nb_errors)) / nb_tests

    def validate_test(data_loader, model):
        nb_errors = 0
        nb_tests = 0
        for i, (images, labels) in enumerate(data_loader):
            output = model(images.view(-1, 3, 32, 32)).argmax(1)
            predictions = []
            for prediction in output.tolist() :
                predictions.append(raw_data.classes[prediction])

            for i in range(len(predictions)) :
                if int(predictions[i]) != int(labels[i]) :
                    nb_errors += 1
            nb_tests += len(images)
        
        return (100*(nb_tests-nb_errors)) / nb_tests

    class LeNet5(nn.Module):
        def __init__(self, dropout=0.0):
            super(LeNet5, self).__init__()
            self.conv1 = nn.Conv2d(3, 9, kernel_size=(5, 5))        # 3 * 9 * 5 * 5 + 9
            self.conv1_2 = nn.Conv2d(9, 9, kernel_size=(3, 3))      # 9 * 9 * 3 * 3 + 9
            self.dropout1 = nn.Dropout2d(p=dropout)
            self.batchnorm1 = nn.BatchNorm2d(9)                     # as attribute, for affine=True

            self.conv2 = nn.Conv2d(9, 32, kernel_size=(4,4))        # 9 * 32 * 3 * 3 + 32
            self.conv2_1 = nn.Conv2d(32, 32, kernel_size=(3, 3))    # 32 * 32 * 3 * 3 + 32
            self.dropout2 = nn.Dropout2d(p=dropout)
            self.batchnorm2 = nn.BatchNorm2d(32)                    # as attribute, for affine=True

            self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3))       # 32 * 64 * 3 * 3 + 92
            self.conv3_1 = nn.Conv2d(64, 64, kernel_size=(3, 3))    # 64 * 64 * 3 * 23 + 92
            self.dropout3 = nn.Dropout(p=dropout)
            self.batchnorm3 = nn.BatchNorm2d(120)                    # as attribute, for affine=True

            self.fc2 = nn.Linear(120, 98)                          # 120 * 98 + 98
            self.batchnorm4 = nn.BatchNorm1d(98)

            self.fc3 = nn.Linear(98, 43)                            # 98 * 43 + 43

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
            layer6_d = self.dropout4(layer6)

            layer7 = F.relu(self.fc2(layer6_d))                              # B * 68
            layer7_bn = self.batchnorm4(layer7)       

            layer8 = self.fc3(layer7_bn)                                        # B * 43
            return layer8

    net = LeNet5()

    total_nb_par = 0
    for p in net.parameters():
        total_nb_par += reduce(lambda x, y: x*y, p.shape, 1)
    print("total nb parameters: ", total_nb_par)

    def train_loop(train_loader, model, loss_map, lr=1e-3, epochs=20, weight_decay=0) :
        history = []    #for monitoring

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model.to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # nous utilisons l'optimiseur Adam vu en cours

        # boucle d'apprentissage
        for epoch in range(epochs) :
            loss_epoch = 0.
            train_acc = 0.

            model.train()

            for images, labels in train_loader :
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = loss_map(output, labels)

                model.zero_grad()
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                train_acc += validate_normal(output, labels)

            # affichage des differents epochs
            model.eval()
            test_acc = validate_test(test_dataloader, model)
            train_acc = 100*train_acc/len(train_loader.dataset)
            history.append(
                {'epoch' : epoch + 1,
                'loss' : loss_epoch,
                'train_acc' : train_acc,
                'valid_acc' : test_acc})
            print(f"epoch : {epoch + 1}/{epochs}, loss = {loss_epoch:.6f}, train_acc = {train_acc}, test_acc = {test_acc}%")
        return history

    net     = LeNet5(0.4)
    history = train_loop(train_loader, net, nn.CrossEntropyLoss(), lr=0.00155, epochs=10, weight_decay=0.008)

    def show_loss(histories, without_first=False) :
        history =  []
        if without_first :
            history = histories[1:]
        else :
            history = histories
        fig, ax = plt.subplots()
        ax.plot([h['epoch'] for h in history], [h['loss'] for h in history], label='loss')
        plt.legend()
        plt.show()

    def show_learning(histories, without_first=False):
        history =  []
        if without_first :
            history = histories[1:]
        else :
            history = histories
        fig, ax = plt.subplots()
        ax.plot([h['epoch'] for h in history], [h['train_acc'] for h in history], label='train accuracy')
        ax.plot([h['epoch'] for h in history], [h['valid_acc'] for h in history], label='test accuracy')
        plt.legend()
        plt.show()
        
    show_loss(history, True)
    show_learning(history, False)

    def get_metrics(data_loader, model):
        y_pred = []
        y_true = []
        for i, (images, labels) in enumerate(data_loader):
            
            output = model(images.view(-1, 3, 32, 32)).argmax(1)
            #print(output)
            predictions = []

            for prediction in output.tolist() :
                predictions.append(raw_data.classes[prediction])
            for i in range(len(predictions)) :
                y_pred.append(int(predictions[i]))
                y_true.append(int(labels[i]))
                
        return y_pred, y_true

    y_pred, y_true = get_metrics(test_dataloader, net)

    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred, target_names=CLASSES.values()))

    # on commence par recolter les donnees et transformer les images en tensors
    raw_data_gr = datasets.ImageFolder("panneaux_route/Train", transform=transforms.Compose(
        [   transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32,32)),
            transforms.ToTensor()
        ]
    ))

    # construisons le dataloader
    train_loader_gr = DataLoader(raw_data_gr, batch_size=256, shuffle=True)

    for images, labels in train_loader_gr :
        for i in range(5) :
            plt.imshow(images[i].permute(1,2,0))
            plt.show()
            print(CLASSES[int(raw_data.classes[labels[i]])]) # pour palier au probleme d'ordre des classes voici ce que nous devons faire
        break

    df = pd.read_csv('panneaux_route/Test.csv', sep=',')
    transformation = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((32,32)), transforms.ToTensor()])

    labels = list(df['ClassId'])
    images = list(df['Path'])
    img_lst = []

    for img in images :
        img = "panneaux_route/" + img
        img_lst.append(transformation(Image.open(img)))

    # le dataset
    test_dataset_gr = MyDataSet(img_lst, labels)

    # le dataloader
    test_dataloader_gr = DataLoader(test_dataset_gr, batch_size=256)

    def validate_test_gr(data_loader, model):
        nb_errors = 0
        nb_tests = 0
        for i, (images, labels) in enumerate(data_loader):
            output = model(images.view(-1, 1, 32, 32)).argmax(1)
            predictions = []
            for prediction in output.tolist() :
                predictions.append(raw_data.classes[prediction])

            for i in range(len(predictions)) :
                if int(predictions[i]) != int(labels[i]) :
                    nb_errors += 1
            nb_tests += len(images)
        
        return (100*(nb_tests-nb_errors)) / nb_tests

    def train_loop_gr(train_loader, model, loss_map, lr=1e-3, epochs=20, weight_decay=0) :
        history = []    #for monitoring

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model.to(device)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay) # nous utilisons l'optimiseur Adam vu en cours

        # boucle d'apprentissage
        for epoch in range(epochs) :
            loss_epoch = 0.
            train_acc = 0.

            model.train()

            for images, labels in train_loader :
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                loss = loss_map(output, labels)

                model.zero_grad()
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                train_acc += validate_normal(output, labels)

            # affichage des differents epochs
            model.eval()
            test_acc = validate_test_gr(test_dataloader_gr, model)
            train_acc = 100*train_acc/len(train_loader.dataset)
            history.append(
                {'epoch' : epoch + 1,
                'loss' : loss_epoch,
                'train_acc' : train_acc,
                'valid_acc' : test_acc})
            print(f"epoch : {epoch + 1}/{epochs}, loss = {loss_epoch:.6f}, train_acc = {train_acc}, test_acc = {test_acc}%")
        return history

    class LeNet5_GR(nn.Module):
        def __init__(self, dropout=0.0):
            super(LeNet5_GR, self).__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5))        # 3 * 9 * 5 * 5 + 9
            self.conv1_2 = nn.Conv2d(16, 22, kernel_size=(5, 5))      # 9 * 9 * 5 * 5 + 9
            self.dropout1 = nn.Dropout2d(p=dropout)
            self.batchnorm1 = nn.BatchNorm2d(22)                     # as attribute, for affine=True

            self.conv2 = nn.Conv2d(22, 32, kernel_size=(3,3))        # 9 * 32 * 3 * 3 + 32
            self.conv2_1 = nn.Conv2d(32, 42, kernel_size=(3, 3))    # 32 * 32 * 3 * 3 + 32
            self.dropout2 = nn.Dropout2d(p=dropout)
            self.batchnorm2 = nn.BatchNorm2d(42)                    # as attribute, for affine=True

            self.conv3 = nn.Conv2d(42, 92, kernel_size=(3,3))       # 32 * 92 * 3 * 3 + 92
            self.conv3_1 = nn.Conv2d(92, 128, kernel_size=(2, 2))    # 92 * 92 * 2 * 2 + 92
            self.dropout3 = nn.Dropout(p=dropout)
            self.batchnorm3 = nn.BatchNorm2d(128)                    # as attribute, for affine=True

            self.fc1 = nn.Linear(128, 240)                           # 120 * 120 + 120
            self.dropout4 = nn.Dropout(p=dropout)

            self.fc2 = nn.Linear(240, 92)                           # 120 * 92 + 92
            self.fc2_1 = nn.Linear(92, 68)                          # 92 * 68 + 68
            self.batchnorm4 = nn.BatchNorm1d(68)

            self.fc3 = nn.Linear(68, 43)                            # 68 * 43 + 43

        def forward(self, input):                                               # B * 3 * 32 * 32  
            layer1 = F.relu(self.conv1(input))                                  # B * 6 * 28 * 28    28 car T-K+1 : (32 - 5 + 1) * (32 - 5 + 1)
            layer1_2 = F.relu(self.conv1_2(layer1))                             # B * 9 * 24 * 24
            layer2 = F.max_pool2d(layer1_2, kernel_size=(2, 2), stride=2)       # B * 6 * 12 * 12
            layer2_d = self.dropout1(layer2)
            layer2_bn = self.batchnorm1(layer2_d)

            layer3 = F.relu(self.conv2(layer2_bn))                              # B * 16 * 10 * 10
            layer3_1 = F.relu(self.conv2_1(layer3))                             # B * 16 * 8 * 8
            layer4 = F.max_pool2d(layer3_1, kernel_size=(2, 2), stride=2)       # B * 16 * 4 * 4
            layer4_d = self.dropout1(layer4)
            layer4_bn = self.batchnorm2(layer4_d)

            layer5 = F.relu(self.conv3(layer4_bn))                              # B * 120 * 2 * 2
            layer5_1 = F.relu(self.conv3_1(layer5))
            layer5_d = self.dropout3(layer5_1)
            layer5_bn = self.batchnorm3(layer5_d)     

            layer6 = F.relu(self.fc1(torch.flatten(layer5_bn,1)))               # B * 92
            layer6_d = self.dropout4(layer6)

            layer8 = F.relu(self.fc2(layer6_d)) 
            layer7 = F.relu(self.fc2_1(layer8))                                 # B * 68
            layer8_bn = self.batchnorm4(layer7)       

            layer7 = self.fc3(layer8_bn)                                        # B * 43
            return layer7
        
    net_gr = LeNet5_GR()

    total_nb_par = 0
    for p in net_gr.parameters():
        total_nb_par += reduce(lambda x, y: x*y, p.shape, 1)
    print("total nb parameters: ", total_nb_par)

    net_gr     = LeNet5_GR(0.4)
    history = train_loop_gr(train_loader_gr, net_gr, nn.CrossEntropyLoss(), lr=0.0018, epochs=15, weight_decay=0.01)
    show_loss(history)
    show_learning(history)

    def get_metrics_gr(data_loader, model):
        y_pred = []
        y_true = []
        for i, (images, labels) in enumerate(data_loader):
            
            output = model(images.view(-1, 1, 32, 32)).argmax(1)
            #print(output)
            predictions = []

            for prediction in output.tolist() :
                predictions.append(raw_data.classes[prediction])
            for i in range(len(predictions)) :
                y_pred.append(int(predictions[i]))
                y_true.append(int(labels[i]))
                
        return y_pred, y_true
    y_pred, y_true = get_metrics_gr(test_dataloader_gr, net_gr)
    print(classification_report(y_true, y_pred, target_names=CLASSES.values()))