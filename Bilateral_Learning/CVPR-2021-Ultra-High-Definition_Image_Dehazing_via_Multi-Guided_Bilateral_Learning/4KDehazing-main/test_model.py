import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import network
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torch 
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from torchvision.utils import save_image
import network


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

my_model = network.B_transformer().to(device)
my_model.eval()
my_model.to(device)


my_model.load_state_dict(torch.load("./model/4K_ohaze.pth")) 
#GAN.load_state_dict(torch.load("/home/dell/IJCAI/JBL/JBPSC/model/model_g_epoch69.pth"))
to_pil_image = transforms.ToPILImage()


tfs_full = transforms.Compose([
            #transforms.Resize(1080),
            transforms.ToTensor()
        ])



def load_simple_list(src_path):
    name_list = list()
    for name in os.listdir(src_path):
        path = os.path.join(src_path, name)
        name_list.append(path)
    name_list = [name for name in name_list if '.jpg' in name]
    name_list.sort()
    return name_list
   
testDataPath = '../../Dehazing_data/test_1000/haze'
list_s = load_simple_list(testDataPath)


i = 0

for idx in range(len(list_s)):
     path = os.path.join(testDataPath, list_s[idx])
     image_in = Image.open(path).convert('RGB')

     full = tfs_full(image_in).unsqueeze(0).to(device)
     beginTime = time.time()
     output = my_model(full)
     

     save_image(output[0], 'test_result/{}.jpg'.format('27_outdoor_hazy'))


