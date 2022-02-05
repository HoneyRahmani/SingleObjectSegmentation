# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 11:12:54 2021

@author: asus
"""

# =========================Data exploration
### Access to images and masks List
import os
import torch

path2data = "./data/training_set/"

imgsList = [pp for pp in os.listdir(path2data) if "Annotation" 
            not in pp]
anntsList = [pp for pp in os.listdir(path2data) if "Annotation"
            in pp]

#print("Length of Images List",len(imgsList))
#print("Length of Annotation List", len(anntsList))

### Display images and masks

import numpy as np
np.random.seed(2019)
rndImg = np.random.choice(imgsList,4)
#print(rndImg)

import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage as ndi
from skimage.segmentation import mark_boundaries

def show_img_mask(img, mask):
    
    if torch.is_tensor(img):
        img = to_pil_image(img)
        mask = to_pil_image(mask)
        
    img_mask = mark_boundaries(np.array(img), np.array(mask)
                               ,outline_color=(0,1,0)
                               , color = (0,1,0))
    plt.imshow(img_mask)


# =============================================================================
# for fn in rndImg:
#     
#     path2img = os.path.join(path2data,fn)
#     path2annt = path2img.replace(".png", "_Annotation.png")
#     
#     img = Image.open(path2img)
#     annt_edge = Image.open(path2annt)
#     mask = ndi.binary_fill_holes(annt_edge)
#     
#     plt.figure()
#     plt.subplot(1,3,1)
#     plt.imshow(img, cmap="gray")
# 
#     plt.subplot(1,3,2)
#     plt.imshow(mask, cmap="gray") 
#     
#     plt.subplot(1,3,3)
#     show_img_mask(img, mask)
# =============================================================================

# =========================Data Augmentation 
from albumentations import HorizontalFlip,VerticalFlip,Compose,Resize

h,w = 128,192
transform_train = Compose([Resize(h, w),
                           HorizontalFlip(p=0.5),
                           VerticalFlip(p=0.5),
                            ])

transform_val = Resize(h,w)




# =========================Creating dataset

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor,to_pil_image


class fetal_dataset(Dataset):
    def __init__(self, path2data, transform=None):      

        imgsList=[pp for pp in os.listdir(path2data) if "Annotation" not in pp]
        #anntsList=[pp for pp in os.listdir(path2data) if "Annotation" in pp]

        self.path2imgs = [os.path.join(path2data, fn) for fn in imgsList] 
        self.path2annts= [p2i.replace(".png", "_Annotation.png") for p2i in self.path2imgs]

        self.transform = transform
    
    def __len__(self):
        return len(self.path2imgs)
      
    def __getitem__(self, idx):
        path2img = self.path2imgs[idx]
        image = Image.open(path2img)

        path2annt = self.path2annts[idx]
        annt_edges = Image.open(path2annt)
        mask = ndi.binary_fill_holes(annt_edges)        
        
        image= np.array(image)
        mask=mask.astype("uint8")        

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']            

        image= to_tensor(image)            
        mask=255*to_tensor(mask)            
        return image, mask



### Define two dataset for Train and Validation
fetal_ds1 = fetal_dataset(path2data, transform=transform_train)
fetal_ds2 = fetal_dataset(path2data, transform=transform_val)

#print(len(fetal_ds1))
#print(len(fetal_ds2))

### Fetch a sample Image from database
img,mask = fetal_ds1[0]
#print(img.shape,img.type(),torch.max(img))
#print(mask.shape,mask.type(),torch.max(mask))

show_img_mask(img, mask)
###Spilit data two group, train and validation

from sklearn.model_selection import ShuffleSplit

sss = ShuffleSplit(n_splits=1, test_size=0.2,random_state=0)

indec = range(len(fetal_ds1))

for train_idx, val_idx in sss.split(indec):
    
    print(len(train_idx))
    print(len(val_idx))
    
### Create train database and validation database

from torch.utils.data import Subset

train_ds =  Subset(fetal_ds1, train_idx)
val_ds = Subset(fetal_ds2, val_idx)

plt.figure()
for imgs,masks in train_ds:
    
    show_img_mask(imgs, masks)
    break

plt.figure()
for imgs,masks in val_ds:
    
    show_img_mask(imgs, masks)
    break    
######### Create data loader
from torch.utils.data import DataLoader

train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)

for train_img,train_mask in train_dl:
    print(train_img.shape)
    print(train_mask.shape)
    break
    
for val_img,val_mask in val_dl:
    print(val_img.shape)
    print(val_mask.shape)
    break

# ========================Defining the model

import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):
    
    def __init__(self, params):
        super(SegNet, self).__init__()
        C_in, H_in, w_in = params["input_shape"]
        init_f = params["initial_filters"]
        num_outputs = params ["num_outputs"]
        
        self.conv1 = nn.Conv2d(C_in, init_f, 
        kernel_size=3, stride =1, padding=1)
        
        self.conv2 = nn.Conv2d(init_f, 2*init_f, 
        kernel_size=3, stride =1, padding=1)
        
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, 
        kernel_size=3, stride =1, padding=1)
        
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, 
        kernel_size=3, stride =1, padding=1)
        
        self.conv5 = nn.Conv2d(8*init_f, 16*init_f, 
        kernel_size=3, stride =1, padding=1)
    
        self.upsample = nn.Upsample(scale_factor=2,
        mode='bilinear', align_corners=True)
        
        self.conv_up1 = nn.Conv2d(16*init_f, 8*init_f, 
        kernel_size=3, padding=1)
        
        self.conv_up2 = nn.Conv2d(8*init_f, 4*init_f, 
        kernel_size=3, padding=1)
        
        self.conv_up3 = nn.Conv2d(4*init_f, 2*init_f, 
        kernel_size=3, padding=1)
        
        self.conv_up4 = nn.Conv2d(2*init_f, init_f, 
        kernel_size=3, padding=1)
        
        self.conv_out = nn.Conv2d(init_f, num_outputs, 
        kernel_size=3, padding=1)
        
    
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = F.relu(self.conv5(x))
        
        x = self.upsample(x)
        x = F.relu(self.conv_up1(x))
        
        x = self.upsample(x)
        x = F.relu(self.conv_up2(x))
        
        x = self.upsample(x)
        x = F.relu(self.conv_up3(x))
        
        x = self.upsample(x)
        x = F.relu(self.conv_up4(x))
        
        x = self.conv_out(x)
        return x
    
    
    
    
params_model = {
        "input_shape" :(1,h,w),
        "initial_filters": 16,
        "num_outputs": 1,
        }

model = SegNet(params_model)




device = torch.device('cuda' if torch.cuda.is_available()
                     else 'cpu' )
model = model.to(device)
print(model)        

from torchsummary import summary
summary(model, input_size=(1,h,w), device=device.type) 
   
# ======================Defining the loss function and optimizer
# === Calculate the dice metric:        
def dice_loss (pred, target, smooth = 1e-5):

    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3))+target.sum(dim=(2,3))        
    dice = 2.0 * (intersection + smooth)/(union + smooth)
    loss = 1.0 - dice
    return loss.sum(), dice.sum()

import torch.nn.functional as F

def loss_func(pred, target):
    
    bce = F.binary_cross_entropy_with_logits(pred, target, 
                                             reduction='sum')
    pred = torch.sigmoid(pred)
    dlv, _ = dice_loss(pred, target)
    loss = bce + dlv
    return loss


def metrics_batch(pred, target):
    
    pred = torch.sigmoid(pred)
    _, metric = dice_loss(pred, target)
    return metric

def loss_batch(loss_func, output, target, opt=None):
    
    loss = loss_func(output, target)
    _, metric_b = dice_loss(output, target)
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    
    return loss.item(), metric_b
from torch import optim
opt = optim.Adam(model.parameters(), lr=3e-4)

from torch.optim.lr_scheduler import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, 
                        patience=20, verbose=1)


def loss_epoch(model, loss_func, dataset_dl, sanity_check=False, opt=None) :
    

     running_loss = 0.0
     running_metric = 0.0
      
     len_data = len(dataset_dl.dataset)
      
     for xb, yb in dataset_dl:
          
          
          xb = xb.to(device)
          yb = yb.to(device)
          
          
          output = model(xb)
          loss_b , metric_b = loss_batch(loss_func, output, yb, opt)
          
          running_loss += loss_b
          
          if metric_b is not None:
              running_metric += metric_b
          if sanity_check is True:
              break
          
     loss = running_loss / float(len_data)
     metric = running_metric / float(len_data)
     
     return loss, metric
 
def get_lr(opt):
    
    for params_group in opt.param_groups:
        return params_group['lr']
    
 
import copy
def train_val (model,params):
    
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl  = params["val_dl"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]
    
    loss_history = {
        
        "train" : [],
        "val"  : [],
        
        }
    metric_history = {
        
        "train" : [],
        "val" : []
        
        }
    best_model_wts = copy. deepcopy(model.state_dict())
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        
        current_lr = get_lr(opt)
        print ('Epoch {}/{}, current_lr ={}'.format(epoch, num_epochs-1,current_lr))
        
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, sanity_check, opt)
        
        loss_history["train"]. append(train_loss)
        metric_history["train"].append(train_metric)
        
        
        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl, sanity_check)
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        if val_loss < best_loss:
            
            best_loss = val_loss
            best_model_wts  = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")
            
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print ("Loading best model weight!")
        print("train loss: %.6f, dice: %.2f"
              %(train_loss,100*train_metric))
        print("val loss: %.6f, dice: %.2f"
              %(val_loss,100*val_metric))
        print("-"*10)
    model.load_state_dict(best_model_wts)
    return   model, loss_history, metric_history

#========Call train_val function
path2models = "./models/"
if not os.path.exists(path2models):
    os.mkdir(path2models)
params_train = {
            
        "num_epochs": 10,
        "optimizer" : opt,
        "loss_func" : loss_func,
        "train_dl" : train_dl,
        "val_dl" : val_dl,
        "sanity_check": False,
        "lr_scheduler" : lr_scheduler,
        "path2weights" : path2models+"weights.pt",
    }
model, loss_hist, metric_hist = train_val(model, params_train)
  
# =======Drowing plot  
num_epochs = params_train["num_epochs"]

# Plot Loss
plt.title("Train_Val Loss")  
plt.plot(range(1, num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1, num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss") 
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

#Plot Accuracy
plt.title("Train_Val Accuracy")  
plt.plot(range(1, num_epochs+1),metric_hist["train"],label="train")
plt.plot(range(1, num_epochs+1),metric_hist["val"],label="val")
plt.ylabel("Accuracy") 
plt.xlabel("Training Epochs")
plt.legend()
plt.show()


 
         
    
    
     
     
    
    
    

    
    
    


