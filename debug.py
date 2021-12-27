import argparse,os,glob,cv2,shutil,logging,math
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import RandomCrop, Compose, ToTensor
from torchvision.utils import save_image
from dataset import DatasetFromFolder
from collections import OrderedDict

from module import *
from criterion import *
from util import *
from module import ssim
from ac.torchac import *
from ac import arithmeticcoding

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# Training settings
shape_num=64
opt = argparse.ArgumentParser(description='').parse_args()
opt.cuda = True
opt.seed = 100001431
opt.threads = 4
opt.table_range = 128
opt.patchSize = 512
opt.batchSize = 8
opt.model_prefix = ''
opt.mode = 'test'
opt.norm = "GSDN"
opt.hyperprior = "gauss"
opt.channels = 192
opt.last_channels = 384
opt.hyper_channels = 192
opt.K = 1
# opt.model_pretrained = '../pytorch_TTSR_comp/checkpoint/ballenet2_GSDN2_c192_s384_gauss_K1_refAR82/mse0.3_stage4/model_epoch_3000.pth'
opt.model_pretrained = 'checkpoint/ballenet2_GSDN_c192_s384_z192_gauss3_K1_hyper_ar_ref_debug2/openpart256_pretrain_uniforminit_lr1e-4_mse0.005/model_epoch_200.pth'
opt.num_parameter = 3  #


# Load train set and test set
# train_set = DatasetFromFolder("./data/train", input_transform=Compose([RandomCrop(opt.patchSize), ToTensor()]), cache=False) 
# training_data_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=8, shuffle=True)
test_set = DatasetFromFolder("./data/kodak", input_transform=Compose([ToTensor()]))
# test_set = DatasetFromFolder("./data/train_val_512", input_transform=Compose([ToTensor()]), cache=False)
testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)