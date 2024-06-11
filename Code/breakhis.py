import dualopt, torch, wavemix, torchvision
from wavemix.classification import WaveMix
import torchvision.transforms as transforms
import math
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
from torchmetrics.classification import Accuracy
import numpy as np
from torchinfo import summary
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset
import argparse
#use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Split data into train, test, and validation sets.")

parser.add_argument('-model', type=str, help="Model")
parser.add_argument('-bs', type=int, help="Batch size")

args = parser.parse_args()

num_classes = 2

if args.model == "wavemix":
    model = WaveMix(
        num_classes = 1000,
        depth = 16,
        mult = 2,
        ff_channel = 192,
        final_dim = 192,
        dropout = 0.5,
        level = 3,
        initial_conv = 'pachify',
        patch_size = 4
    )

    url = 'https://huggingface.co/cloudwalker/wavemix/resolve/main/Saved_Models_Weights/ImageNet/wavemix_192_16_75.06.pth'
    model.load_state_dict(torch.hub.load_state_dict_from_url(url))
    print("ImageNet Weights Loaded") 
    model.pool[2] = nn.Linear(192, num_classes)

elif args.model == "swin":
    from torchvision.models import swin_t, Swin_T_Weights
    model = swin_t(weights= Swin_T_Weights.IMAGENET1K_V1)
    model.head = nn.Linear(768, num_classes)

elif args.model == "swinv2":
    from torchvision.models import swin_v2_t, Swin_V2_T_Weights
    model = swin_v2_t(weights= Swin_V2_T_Weights.IMAGENET1K_V1)
    model.head = nn.Linear(768, num_classes)

elif args.model == "efficientnet":
    from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
    model = efficientnet_v2_s(weights= EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(1280, num_classes)

elif args.model == "convnext":
    from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    model = convnext_tiny(weights= ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = nn.Linear(768, num_classes)

elif args.model == "resnet":
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights= ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(2048, num_classes)

elif args.model == "densenet":
    from torchvision.models import densenet161, DenseNet161_Weights
    model = densenet161(weights= DenseNet161_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(2208, num_classes)

elif args.model == "inception":
    from torchvision.models import inception_v3, Inception_V3_Weights
    model = inception_v3(weights= Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(2048, num_classes)

elif args.model == "mobilenet":
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
    model = mobilenet_v3_large(weights= MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model.classifier[3] = nn.Linear(1280, num_classes)

elif args.model == "regnet":
    from torchvision.models import regnet_y_3_2gf, RegNet_Y_3_2GF_Weights
    model = regnet_y_3_2gf(weights= RegNet_Y_3_2GF_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(1512, num_classes)

elif args.model == "resnext":
    from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
    model = resnext50_32x4d(weights= ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(2048, num_classes)

elif args.model == "shufflenet":
    from torchvision.models import shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights
    model = shufflenet_v2_x2_0(weights= ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(2048, num_classes)
else:
    print("model not found")
    exit()

model.to(device)
print(summary(model, (1, 3, 448, 672)))


transform_test = transforms.Compose(
        [transforms.Resize([448, 672]),
         transforms.ToTensor(),
    #  transforms.Normalize((0.8036, 0.6536, 0.7731), (0.1091, 0.1494, 0.1070))  #40x
    #  transforms.Normalize((0.7997, 0.6358, 0.7745), (0.1270, 0.1779, 0.1187))  #100x
     transforms.Normalize((0.7917, 0.6211, 0.7659), (0.1244, 0.1780, 0.1099))    #200x
    #  transforms.Normalize((0.7651, 0.5763, 0.7524), (0.1397, 0.1966, 0.1112))  #400x
     ])

transform_train = transforms.Compose(
        [transforms.Resize([448, 672]),
         transforms.ToTensor(),
    #  transforms.Normalize((0.8018, 0.6498, 0.7656), (0.1105, 0.1555, 0.1126)) #40x
    #  transforms.Normalize((0.7940, 0.6362, 0.7699), (0.1279, 0.1791, 0.1167))  #100x
     transforms.Normalize((0.7895, 0.6215, 0.7696), (0.1266, 0.1784, 0.1077))  #200
    #  transforms.Normalize((0.7562, 0.5913, 0.7426), (0.1425, 0.2017, 0.1156)) #400x
    ])

batch_size = 32

trainset = torchvision.datasets.ImageFolder(root='Breakhis/200x/train', transform=transform_train)
valset = torchvision.datasets.ImageFolder(root='Breakhis/200x/val', transform=transform_test)
testset = torchvision.datasets.ImageFolder(root='Breakhis/200x/test', transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)

testset = ConcatDataset([valset, testset])

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size*2,
                                         shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2, persistent_workers=2)
#define datasets and dataloaders


print(len(trainset))
print(len(testset))



top1_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)


#loss
criterion = nn.CrossEntropyLoss()

#Mixed Precision training
scaler = torch.cuda.amp.GradScaler()

top1 = [0]
traintime = []
testtime = []
counter = 0

# Use AdamW or lion as the first optimizer

optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
print("Training with AdamW")

# # optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
# # print("Training with Lion")

# load saved model
PATH = 'train.pth'
# model.load_state_dict(torch.load(PATH))

epoch = 0
while counter < 10:   #Counter sets the number of epochs of non improvement before stopping

    t0 = time.time()
    epoch_accuracy = 0
    epoch_loss = 0
    running_loss = 0.0
    model.train()

    with tqdm(trainloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")

        for data in tepoch:
  
          inputs, labels = data[0].to(device), data[1].to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
        #   print(outputs.shape)
          with torch.cuda.amp.autocast():
              loss = criterion(outputs, labels)
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()

          acc = (outputs.argmax(dim=1) == labels).float().mean()
          epoch_accuracy += acc / len(trainloader)
          epoch_loss += loss / len(trainloader)
          tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}" )

    correct_1=0
    c = 0
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            correct_1 += top1_acc(outputs, labels)
            c += 1
        
    print(f"Epoch : {epoch+1} - Top 1: {correct_1*100/c:.2f} - Train Time: {t1 - t0:.2f} - Test Time: {time.time() - t1:.2f}\n")

    if float(correct_1*100/c) > float(max(top1)):
        torch.save(model.state_dict(), PATH)
        print(1)
        counter = 0
    top1.append(correct_1*100/c)
    traintime.append(t1 - t0)
    testtime.append(time.time() - t1)
    counter += 1
    epoch += 1
    
    

# Second Optimizer
print('Training with SGD')

model.load_state_dict(torch.load(PATH))
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

counter = 0
epoch = 0
while counter < 10: # loop over the dataset multiple times
    t0 = time.time()
    epoch_accuracy = 0
    epoch_loss = 0
    running_loss = 0.0
    model.train()

    with tqdm(trainloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch+1}")

        for data in tepoch:
  
          inputs, labels = data[0].to(device), data[1].to(device)
          optimizer.zero_grad()
          outputs = model(inputs)
          with torch.cuda.amp.autocast():
              loss = criterion(outputs, labels)
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()
          
          acc = (outputs.argmax(dim=1) == labels).float().mean()
          epoch_accuracy += acc / len(trainloader)
          epoch_loss += loss / len(trainloader)
          tepoch.set_postfix_str(f" loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}" )

    correct_1=0
    
    c = 0
    model.eval()
    t1 = time.time()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            correct_1 += top1_acc(outputs, labels)
            
            c += 1
        
    print(f"Epoch : {epoch+1} - Top 1: {correct_1*100/c:.2f}  -  Train Time: {t1 - t0:.2f} - Test Time: {time.time() - t1:.2f}\n")

    if float(correct_1*100/c) > float(max(top1)):
        torch.save(model.state_dict(), PATH)
        print(1)
        counter = 0

    top1.append(correct_1*100/c)
    
    traintime.append(t1 - t0)
    testtime.append(time.time() - t1)
    counter += 1
    epoch += 1
    
        
print('Finished Training')
print("Results")
print(f"Top 1 Accuracy: {max(top1):.2f}  - Train Time: {min(traintime):.0f} -Test Time: {min(testtime):.0f}\n")

model.load_state_dict(torch.load(PATH))
def class_weighted_accuracy(predicted_labels, true_labels):
    # Compute the class weights
    class_counts = torch.bincount(true_labels)

    class_weights = 1.0 / class_counts.float()


    # Compute the sample weights
    sample_weights = class_weights[true_labels]


    # Compute the accuracy
    predicted_classes = predicted_labels.argmax(dim=1)

    correct_predictions = (predicted_classes == true_labels).float()
    accuracy = torch.mean(correct_predictions)

    # Compute the class-weighted accuracy
    weighted_accuracy = torch.sum(sample_weights * correct_predictions) / torch.sum(sample_weights)

    return weighted_accuracy.item()


correct_1=0
c = 0
model.eval()
t1 = time.time()
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        correct_1 += class_weighted_accuracy(outputs, labels)
        # correct_1 += multiclass_accuracy(outputs, labels, num_classes=2, average= 'weighted')
        c += 1

print(f" class -weighted acc : {correct_1*100/c:.2f} - Test Time: {time.time() - t1:.2f}\n")
