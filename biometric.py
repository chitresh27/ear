import os
# import albumentations as A
import collections
import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone import Backbone
from tqdm import tqdm
# from albumentations.pytorch import ToTensorV2
from PIL import Image
import pickle
import torch
from torch2trtmaster.torch2trt import torch2trt
import glob
import os
import jwt
import pymysql
import datetime
from datetime import date
from flask import jsonify
from flask import flash, request
from flask_cors import CORS, cross_origin
from flask import Flask,render_template,Response
#from werkzeug import generate_password_hash, check_password_hash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
# from detect_ocr_final import run
# from track_api import detect
import random
# import requests
# from sqlalchemy import create_engine
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import json
# from flask_cors import CORS
from flask import request
import cv2
# from flask_cors import CORS, cross_origin
from db import mysql

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
	AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple
from urllib.parse import urlparse
import json
# from flask_cors import CORS
from flask import request
import cv2
from flask_cors import CORS, cross_origin

from urllib.parse import urlparse

from flask import Flask,render_template,Response

from flask import jsonify

from app import app
import os
import timm
from ultralytics import YOLO
import re
import face_detection
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

detector = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
	AdaptiveAvgPool2d, Sequential, Module
from collections import namedtuple

ear_detection_model = YOLO('ear_detection_yolov8_l.pt', task="detect")

############################ ear detection ################

class BasicConv2d(nn.Module):

	def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
		super().__init__()
		self.conv = nn.Conv2d(
			in_planes, out_planes,
			kernel_size=kernel_size, stride=stride,
			padding=padding, bias=False
		) # verify bias false
		self.bn = nn.BatchNorm2d(
			out_planes,
			eps=0.001, # value found in tensorflow
			momentum=0.1, # default pytorch value
			affine=True
		)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x


# In[8]:


class Block31_down(nn.Module):

	def __init__(self, scale=1.0, noReLU=False):
		super().__init__()

		self.scale = scale
		self.noReLU = noReLU

		self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)
		self.branch1= BasicConv2d(1792, 192, kernel_size=1, stride=1)
		self.branch2= BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1))
		self.branch3= BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
		

#         self.branch1 = nn.Sequential(
#             BasicConv2d(1792, 192, kernel_size=1, stride=1),
#             BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
#             BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
#         )

		self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
		if not self.noReLU:
			self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		x1 = self.branch2(x1)
		x1 = self.branch3(x1)
		out = torch.cat((x0, x1), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		if not self.noReLU:
			out = self.relu(out)
		return out


# In[9]:


class Block35_down(nn.Module):

	def __init__(self, scale=1.0):
		super().__init__()

		self.scale = scale

		self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)
		
		self.branch1_1= BasicConv2d(256, 32, kernel_size=1, stride=1)
		self.branch1_2= BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

#         self.branch1 = nn.Sequential(
#             BasicConv2d(256, 32, kernel_size=1, stride=1),
#             BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         )
		
		self.branch2_1=BasicConv2d(256, 32, kernel_size=1, stride=1)
		self.branch2_2=BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.branch2_3=BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

#         self.branch2 = nn.Sequential(
#             BasicConv2d(256, 32, kernel_size=1, stride=1),
#             BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         )

		self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1_1(x)
		x1 = self.branch1_2(x1)
		x2 = self.branch2_1(x)
		x2 = self.branch2_2(x2)
		x2 = self.branch2_3(x2)
		out = torch.cat((x0, x1, x2), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		out = self.relu(out)
		return out


# In[10]:


class Block17_down(nn.Module):

	def __init__(self, scale=1.0):
		super().__init__()

		self.scale = scale

		self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)
		self.branch1_1=BasicConv2d(896, 128, kernel_size=1, stride=1)
		self.branch1_2=BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3))
		self.branch1_3=BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))

#         self.branch1 = nn.Sequential(
#             BasicConv2d(896, 128, kernel_size=1, stride=1),
#             BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
#             BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
#         )

		self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch0(x)
		
		x1 = self.branch1_1(x)
		x1 = self.branch1_2(x1)
		x1 = self.branch1_3(x1)
		out = torch.cat((x0, x1), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		out = self.relu(out)
		return out


# In[11]:


class Block8_down(nn.Module):

	def __init__(self, scale=1.0, noReLU=False):
		super().__init__()

		self.scale = scale
		self.noReLU = noReLU

		self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1,padding=0)
		self.branch1_1=BasicConv2d(1792, 192, kernel_size=1, stride=1,padding=0)
		self.branch1_2=BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1))
		self.branch1_3=BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
		

#         self.branch1 = nn.Sequential(
#             BasicConv2d(1792, 192, kernel_size=1, stride=1),
#             BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
#             BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
#         )

		self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1,padding=0)
		if not self.noReLU:
			self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1_1(x)
		x1 = self.branch1_2(x1)
		x1 = self.branch1_3(x1)
		out = torch.cat((x0, x1), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		if not self.noReLU:
			out = self.relu(out)
		return out


# In[12]:


class Mixed_6a_down(nn.Module):

	def __init__(self):
		super().__init__()

		self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2,padding=1)
		self.branch1_1=BasicConv2d(256, 192, kernel_size=1, stride=1,padding=0)
		self.branch1_2=BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1)
		self.branch1_3=BasicConv2d(192, 256, kernel_size=3, stride=2,padding=1)

#         self.branch1 = nn.Sequential(
#             BasicConv2d(256, 192, kernel_size=1, stride=1),
#             BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
#             BasicConv2d(192, 256, kernel_size=3, stride=2)
#         )

		self.branch2 = nn.MaxPool2d(3, stride=2,padding=1)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1_1(x)
		x1 = self.branch1_2(x1)
		x1 = self.branch1_3(x1)
		x2 = self.branch2(x)
#         print(x.shape)
#         print(x0.shape)
#         print(x1.shape)
#         print(x2.shape)
		out = torch.cat((x0, x1, x2), 1)
		return out


# In[13]:


class Mixed_7a_down(nn.Module):

	def __init__(self):
		super().__init__()
		
		self.branch0_1=BasicConv2d(896, 256, kernel_size=1, stride=1,padding=0)
		self.branch0_2=BasicConv2d(256, 384, kernel_size=3, stride=2,padding=1)
		

#         self.branch0 = nn.Sequential(
#             BasicConv2d(896, 256, kernel_size=1, stride=1),
#             BasicConv2d(256, 384, kernel_size=3, stride=2)
#         )
		self.branch1_1=BasicConv2d(896, 256, kernel_size=1, stride=1,padding=0)
		self.branch1_2=BasicConv2d(256, 256, kernel_size=3, stride=2,padding=1)


#         self.branch1 = nn.Sequential(
#             BasicConv2d(896, 256, kernel_size=1, stride=1),
#             BasicConv2d(256, 256, kernel_size=3, stride=2)
#         )
		
		self.branch2_1=BasicConv2d(896, 256, kernel_size=1, stride=1,padding=0)
		self.branch2_2=BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1)
		self.branch2_3=BasicConv2d(256, 256, kernel_size=3, stride=2,padding=1)

#         self.branch2 = nn.Sequential(
#             BasicConv2d(896, 256, kernel_size=1, stride=1),
#             BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             BasicConv2d(256, 256, kernel_size=3, stride=2)
#         )

		self.branch3 = nn.MaxPool2d(3, stride=2,padding=1)

	def forward(self, x):
		x0 = self.branch0_1(x)
		x0 = self.branch0_2(x0)
		x1 = self.branch1_1(x)
		x1 = self.branch1_2(x1)
		x2 = self.branch2_1(x)
		x2 = self.branch2_2(x2)
		x2 = self.branch2_3(x2)
		x3 = self.branch3(x)
		out = torch.cat((x0, x1, x2, x3), 1)
		return out


# In[14]:


class BasicConvtranspose2d(nn.Module):

	def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
		super().__init__()
		self.conv = nn.ConvTranspose2d(
			in_planes, out_planes,
			kernel_size=kernel_size, stride=stride,
			padding=padding, bias=False
		) # verify bias false
		self.bn = nn.BatchNorm2d(
			out_planes,
			eps=0.001, # value found in tensorflow
			momentum=0.1, # default pytorch value
			affine=True
		)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x


# In[15]:


class Block31_up(nn.Module):

	def __init__(self, scale=1.0, noReLU=False):
		super().__init__()

		self.scale = scale
		self.noReLU = noReLU

		self.branch0 = BasicConvtranspose2d(1792, 192, kernel_size=1, stride=1)
		self.branch1= BasicConvtranspose2d(1792, 192, kernel_size=1, stride=1)
		self.branch2= BasicConvtranspose2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1))
		self.branch3= BasicConvtranspose2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
		

#         self.branch1 = nn.Sequential(
#             BasicConv2d(1792, 192, kernel_size=1, stride=1),
#             BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
#             BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
#         )

		self.conv2d = nn.ConvTranspose2d(384, 1792, kernel_size=1, stride=1)
		if not self.noReLU:
			self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		x1 = self.branch2(x1)
		x1 = self.branch3(x1)
		out = torch.cat((x0, x1), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		if not self.noReLU:
			out = self.relu(out)
		return out


# In[16]:


class Block35_up(nn.Module):

	def __init__(self, scale=1.0):
		super().__init__()

		self.scale = scale

		self.branch0 = BasicConvtranspose2d(256, 32, kernel_size=1, stride=1,padding=0)
		
		self.branch1_1=BasicConvtranspose2d(256, 32, kernel_size=1, stride=1,padding=0)
		self.branch1_2=BasicConvtranspose2d(32, 32, kernel_size=3, stride=1, padding=1)

#         self.branch1 = nn.Sequential(
#             BasicConv2d(256, 32, kernel_size=1, stride=1),
#             BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         )
		
		self.branch2_1=BasicConvtranspose2d(256, 32, kernel_size=1, stride=1,padding=0)
		self.branch2_2=BasicConvtranspose2d(32, 32, kernel_size=3, stride=1, padding=1)
		self.branch2_3=BasicConvtranspose2d(32, 32, kernel_size=3, stride=1, padding=1)

#         self.branch2 = nn.Sequential(
#             BasicConv2d(256, 32, kernel_size=1, stride=1),
#             BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
#             BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
#         )

		self.conv2d = nn.ConvTranspose2d(96, 256, kernel_size=1, stride=1,padding=0)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1_1(x)
		x1 = self.branch1_2(x1)
		x2 = self.branch2_1(x)
		x2 = self.branch2_2(x2)
		x2 = self.branch2_3(x2)
		out = torch.cat((x0, x1, x2), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		out = self.relu(out)
		return out


# In[17]:


class Block17_up(nn.Module):

	def __init__(self, scale=1.0):
		super().__init__()

		self.scale = scale

		self.branch0 = BasicConvtranspose2d(896, 128, kernel_size=1, stride=1)
		self.branch1_1=BasicConvtranspose2d(896, 128, kernel_size=1, stride=1)
		self.branch1_2=BasicConvtranspose2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3))
		self.branch1_3=BasicConvtranspose2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))

#         self.branch1 = nn.Sequential(
#             BasicConv2d(896, 128, kernel_size=1, stride=1),
#             BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
#             BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
#         )

		self.conv2d = nn.ConvTranspose2d(256, 896, kernel_size=1, stride=1)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1_1(x)
		x1 = self.branch1_2(x1)
		x1 = self.branch1_3(x1)
		out = torch.cat((x0, x1), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		out = self.relu(out)
		return out


# In[18]:


class Block8_up(nn.Module):

	def __init__(self, scale=1.0, noReLU=False):
		super().__init__()

		self.scale = scale
		self.noReLU = noReLU

		self.branch0 = BasicConvtranspose2d(1792, 192, kernel_size=1, stride=1,padding=0)
		self.branch1_1=BasicConvtranspose2d(1792, 192, kernel_size=1, stride=1,padding=0)
		self.branch1_2=BasicConvtranspose2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1))
		self.branch1_3=BasicConvtranspose2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
		

#         self.branch1 = nn.Sequential(
#             BasicConv2d(1792, 192, kernel_size=1, stride=1),
#             BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
#             BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
#         )

		self.conv2d = nn.ConvTranspose2d(384, 1792, kernel_size=1, stride=1,padding=0)
		if not self.noReLU:
			self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1_1(x)
		x1 = self.branch1_2(x1)
		x1 = self.branch1_3(x1)
		out = torch.cat((x0, x1), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		if not self.noReLU:
			out = self.relu(out)
		return out


# In[19]:


class Mixed_6a_up(nn.Module):

	def __init__(self):
		super().__init__()

		self.branch0 = BasicConvtranspose2d(896, 128, kernel_size=3, stride=1,padding=1)
		self.branch1_1=BasicConvtranspose2d(896, 192, kernel_size=1, stride=1,padding=0)
		self.branch1_2=BasicConvtranspose2d(192, 192, kernel_size=3, stride=1, padding=1)
		self.branch1_3=BasicConvtranspose2d(192, 128, kernel_size=3, stride=1,padding=1)

#         self.branch1 = nn.Sequential(
#             BasicConv2d(256, 192, kernel_size=1, stride=1),
#             BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
#             BasicConv2d(192, 256, kernel_size=3, stride=2)
#         )

		self.branch2 = nn.Upsample(scale_factor=2, mode='bilinear')

	def forward(self, x):
		x0 = self.branch0(x)
		x0 = self.branch2(x0)
		x1 = self.branch1_1(x)
		x1 = self.branch1_2(x1)
		x1 = self.branch1_3(x1)
		x1 = self.branch2(x1)
#         x2 = self.branch2(x)
#         print(x.shape)
#         print(x0.shape)
#         print(x1.shape)
#         print(x2.shape)
		out = torch.cat((x0, x1), 1)
		return out


# In[20]:


class Mixed_7a_up(nn.Module):

	def __init__(self):
		super().__init__()
		
		self.branch0_1=BasicConvtranspose2d(1792, 256, kernel_size=1, stride=1,padding=0)
		self.branch0_2=BasicConvtranspose2d(256, 384, kernel_size=3, stride=1,padding=1)
		

#         self.branch0 = nn.Sequential(
#             BasicConv2d(896, 256, kernel_size=1, stride=1),
#             BasicConv2d(256, 384, kernel_size=3, stride=2)
#         )
		self.branch1_1=BasicConvtranspose2d(1792, 256, kernel_size=1, stride=1,padding=0)
		self.branch1_2=BasicConvtranspose2d(256, 256, kernel_size=3, stride=1,padding=1)


#         self.branch1 = nn.Sequential(
#             BasicConv2d(896, 256, kernel_size=1, stride=1),
#             BasicConv2d(256, 256, kernel_size=3, stride=2)
#         )
		
		self.branch2_1=BasicConvtranspose2d(1792, 256, kernel_size=1, stride=1,padding=0)
		self.branch2_2=BasicConvtranspose2d(256, 256, kernel_size=3, stride=1, padding=1)
		self.branch2_3=BasicConvtranspose2d(256, 256, kernel_size=3, stride=1,padding=1)

#         self.branch2 = nn.Sequential(
#             BasicConv2d(896, 256, kernel_size=1, stride=1),
#             BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             BasicConv2d(256, 256, kernel_size=3, stride=2)
#         )

		self.branch3 = nn.Upsample(scale_factor=2, mode='bilinear')

	def forward(self, x):
		x0 = self.branch0_1(x)
		x0 = self.branch0_2(x0)
		x0 = self.branch3(x0)
		x1 = self.branch1_1(x)
		x1 = self.branch1_2(x1)
		x1 = self.branch3(x1)
		x2 = self.branch2_1(x)
		x2 = self.branch2_2(x2)
		x2 = self.branch2_3(x2)
		x2 = self.branch3(x2)
#         x3 = self.branch3(x)
#         print(x0.shape)
#         print(x1.shape)
#         print(x2.shape)
#         print(x3.shape)
		out = torch.cat((x0, x1, x2), 1)
		return out


# In[21]:


class UNET_RESNET(nn.Module):
	"""Inception Resnet V1 model with optional loading of pretrained weights.
	Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
	datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
	requested and cached in the torch cache. Subsequent instantiations use the cache rather than
	redownloading.
	Keyword Arguments:
		pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
			(default: {None})
		classify {bool} -- Whether the model should output classification probabilities or feature
			embeddings. (default: {False})
		num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
			equal to that used for the pretrained model, the final linear layer will be randomly
			initialized. (default: {None})
		dropout_prob {float} -- Dropout probability. (default: {0.6})
	"""
	def __init__(self,dropout_prob=0.6):
		super().__init__()

		# Define layers
		self.OUTPUT= BasicConv2d(3, 1, kernel_size=3, stride=1,padding=1)
		self.upsamle = nn.Upsample(scale_factor=2, mode='bilinear')
		self.conv2d_1a_down = BasicConv2d(3, 32, kernel_size=3, stride=2,padding=1)
		self.conv2d_2a_down = BasicConv2d(32, 32, kernel_size=3, stride=1,padding=1)
		self.conv2d_2b_down =  BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.maxpool_3a_down = nn.MaxPool2d(3, stride=2,padding=1)
		self.conv2d_3b_down = BasicConv2d(64, 80, kernel_size=1, stride=1)
		self.conv2d_4a_down = BasicConv2d(80, 192, kernel_size=3, stride=1,padding=1)
		self.conv2d_4b_down = BasicConv2d(192, 256, kernel_size=3, stride=2,padding=1)
		
		self.conv2d_1a_up = BasicConvtranspose2d( 32,3, kernel_size=3, stride=1,padding=1)
		self.conv2d_2a_up = BasicConvtranspose2d( 32,32, kernel_size=3, stride=1,padding=1)
		self.conv2d_2b_up = BasicConvtranspose2d( 64,32, kernel_size=3, stride=1, padding=1)
#         self.maxpool_3a_up = nn.MaxPool2d(3, stride=2)
		self.conv2d_3b_up = BasicConvtranspose2d( 80,64, kernel_size=1, stride=1,padding=0)
		self.conv2d_4a_up = BasicConvtranspose2d(192,80, kernel_size=3, stride=1,padding=1)
		self.conv2d_4b_up = BasicConvtranspose2d(256,192, kernel_size=3, stride=1,padding=1)
		
		self.repeat_1_5_down =Block35_down(scale=0.17)

		self.mixed_6a__down = Mixed_6a_down()
		self.repeat_2_10__down = Block17_down(scale=0.10)

		self.mixed_7a_down = Mixed_7a_down()
		self.repeat_3_5__down = Block8_down(scale=0.20)

		self.block8_down = Block8_down(noReLU=True)

		
####

		self.repeat_1_5_up =Block35_up(scale=0.17)

		self.mixed_6a__up = Mixed_6a_up()
		self.repeat_2_10__up = Block17_up(scale=0.10)

		self.mixed_7a_up = Mixed_7a_up()
		self.repeat_3_5__up = Block8_up(scale=0.20)

		self.block8_up = Block8_up(noReLU=True)
####
	
		self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
		self.dropout = nn.Dropout(dropout_prob)
		self.last_linear = nn.Linear(1792, 512, bias=False)
		self.last_bn = nn.BatchNorm1d(512)
		
		self.last_linear_up = nn.Linear(512, 1792, bias=False)


	def forward(self, x):
		"""Calculate embeddings or logits given a batch of input image tensors.
		Arguments:
			x {torch.tensor} -- Batch of image tensors representing faces.
		Returns:
			torch.tensor -- Batch of embedding vectors or multinomial logits.
		"""
		x = self.conv2d_1a_down(x)
#         print(x.shape)
		x = self.conv2d_2a_down(x)
#         print(x.shape)
		x = self.conv2d_2b_down(x)
#         print(x.shape)
		x = self.maxpool_3a_down(x)

		x = self.conv2d_3b_down(x)
#         print('0',x.shape)
		x = self.conv2d_4a_down(x)
#         print(x.shape)
		x1 = self.conv2d_4b_down(x)
	
#################################################################

		x_1 = self.repeat_1_5_down(x1)
#         print(x_1.shape)
		x_1 = self.repeat_1_5_down(x_1)
		x_1 = self.repeat_1_5_down(x_1)
		x_1 = self.repeat_1_5_down(x_1)
		x2 = self.repeat_1_5_down(x_1)
######################################################################
#         print('1',x2.shape)
		x3= self.mixed_6a__down(x2)
#         print('2',x3.shape)
######################################################################
		x_3 = self.repeat_2_10__down(x3)
		x_3 = self.repeat_2_10__down(x_3)
		x_3 = self.repeat_2_10__down(x_3)
		x_3 = self.repeat_2_10__down(x_3)
		x_3 = self.repeat_2_10__down(x_3)
		x_3 = self.repeat_2_10__down(x_3)
		x_3 = self.repeat_2_10__down(x_3)
		x_3 = self.repeat_2_10__down(x_3)
		x_3 = self.repeat_2_10__down(x_3)
		x4 = self.repeat_2_10__down(x_3)
		
######################################################
#         print('3',x4.shape)
		x5 = self.mixed_7a_down(x4)
#         print('4',x5.shape)
######################################################
		x_5 = self.repeat_3_5__down(x5)
		x_5 = self.repeat_3_5__down(x_5)
		x_5 = self.repeat_3_5__down(x_5)
		x_5 = self.repeat_3_5__down(x_5)
		x6 = self.repeat_3_5__down(x_5)

########################################################
		
#         print('5',x6.shape)
		x7 = self.block8_down(x6)
################################################
#         print('6',x7.shape)
		x8 = self.avgpool_1a(x7)
#         print('7',x8.shape)
		x8 = self.dropout(x8)
		x8 = self.last_linear(x8.view(x.shape[0], -1))
		x8 = self.last_bn(x8)
		
		x9=self.last_linear_up(x8)
		x9 = self.dropout(x9)
		x9=x9[ :, :, None, None]
		x9 = torch.cat((x9,x9,x9,x9,x9,x9,x9,x9), 2)
		x9 = torch.cat((x9,x9,x9,x9,x9,x9,x9,x9), 3)
		
#         print('8',x9.shape)
		
##################################################


		x9=x9+x7
		
		
		x10 = self.block8_up(x9)
		
#         print('9',x10.shape)
################################################  
		x10=x10+x6
		x_10 = self.repeat_3_5__up(x10)
		x_10 = self.repeat_3_5__up(x_10)
		x_10 = self.repeat_3_5__up(x_10)
		x_10 = self.repeat_3_5__up(x_10)
		x11 = self.repeat_3_5__up(x_10)
		
#         print(x11.shape)
#########################################
		x11=x11+x5
		x12 = self.mixed_7a_up(x11)
		
#         print(x12.shape)
		
##################################################
		x12=x12+x4
		
		
		x_12 = self.repeat_2_10__up(x12)
		x_12 = self.repeat_2_10__up(x_12)
		x_12 = self.repeat_2_10__up(x_12)
		x_12 = self.repeat_2_10__up(x_12)
		x_12 = self.repeat_2_10__up(x_12)
		x_12 = self.repeat_2_10__up(x_12)
		x_12 = self.repeat_2_10__up(x_12)
		x_12 = self.repeat_2_10__up(x_12)
		x_12 = self.repeat_2_10__up(x_12)
		x13 = self.repeat_2_10__up(x_12)
####################################################  

		x13=x13+x3
#         print(x13.shape)
		
		x14= self.mixed_6a__up(x13)
		
##############################################

		x14=x14+x2
		
		
		x_14 = self.repeat_1_5_up(x14)

		x_14 = self.repeat_1_5_up(x_14)
		x_14 = self.repeat_1_5_up(x_14)
		x_14 = self.repeat_1_5_up(x_14)
		x15 = self.repeat_1_5_up(x_14)
###############################################

		x15+x1
#         print('10',x15.shape)
		
		x16 = self.conv2d_4b_up(x15)
		x16=self.upsamle(x16)

		x16 = self.conv2d_4a_up(x16)

		x16 = self.conv2d_3b_up(x16)
		x16=self.upsamle(x16)
		x16 = self.conv2d_2b_up(x16)
		x16 = self.conv2d_2a_up(x16)
		
#         print('111',x16.shape)
		
		
		x17 = self.conv2d_1a_up(x16)
		x17=self.upsamle(x17)
		
		output=self.OUTPUT(x17)
		

		
		
		
		
		
	   
		
		
		

		return output
		
		
		


# In[22]:


# x = torch.randn((1,3,256, 256))
# model = UNET_RESNET()


# # In[23]:


# class keypoint(nn.Module):
# 	def __init__(self):
# 		super(keypoint, self).__init__()
# 		self.cnn1=model
# 		self.flatten =nn.Flatten()
# #       self.fc1 = nn.Sequential(nn.Linear(6400,526),nn.ReLU(),)
# #       self.fc2 = nn.Sequential(nn.Linear(526,126),nn.ReLU(),)
# 		self.fc3 = nn.Sequential(nn.Linear(65536,12))
# #       self.fc4 = nn.Sequential(nn.ReLU())
		

# #       self.fc3 = nn.Sequential(nn.Sigmoid())

# #   def forward_once(self, x):
# #       # Forward pass 
# #       output = self.cnn1(x)
# #       output = output.view(output.size()[0], -1)
# #       output = self.fc1(output)
# # #         output = self.fc2(output)
# #       return output

# 	def forward(self,input1):
# 		output = self.cnn1(input1)
# 		output = self.flatten(output)
# #       output = self.fc1(output)

# #       output = self.fc2(output)
# 		output = self.fc3(output)
# #       output = self.fc4(output)
# #       output_1=output_1[:,0:64]

# 		return output


# net=keypoint().to(torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))


# # In[29]:


# checkpoint = torch.load('keypoint_model_val_30march_opt.pt')
# net.load_state_dict(checkpoint['model_state_dict'])




# with torch.no_grad():
# 	net=net.eval().cuda()
# 	net=net.to(torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))


# # In[30]:


# from torch2trt import torch2trt


# # In[31]:


# model_trt_ear_detect = torch2trt(net, [x.to(torch.device("cuda:1" if torch.cuda.is_available() else "cpu"))])

@app.route('/register', methods=['POST'])
@cross_origin()
def add_data():
	conn = None
	cursor = None
	try:
		print("request ",request.json),
		_json = request.json
		print("request ",_json)
		_password=_json['password']
		_mobileno=_json['mobile_no']
		_site_id=_json['site_id']
		# validate the received values


		


		try:
			data_user=pd.read_csv("user.csv")
		except Exception as e:
			data_init = {
			  "mobileno": [],
			  "password": [],
			  "site_id":[]
			}

			#load data into a DataFrame object:
			data_user = pd.DataFrame(data_init)


		mobileno_list=data_user["mobileno"]
		password_list=data_user["password"]
		site_id_list=data_user["site_id"]


		if _mobileno and _password and request.method == 'POST':
			# conn = mysql.connect()
			# cursor = conn.cursor(pymysql.cursors.DictCursor)

			if _mobileno in mobileno_list:

				msg={"success":"false","message":"user registration failled number already exist"}
				resp = jsonify(msg)
				return resp

			# if cursor.execute("SELECT * FROM add_user WHERE mobileno=%s AND password=%s",(_mobileno, _password,)):
			# 	msg={"success":"false","message":"user registration failled."}
			# 	resp = jsonify(msg)
			# 	return resp
			else:

				password_list.append(_password)
				mobileno_list.append(_mobileno)
				site_id_list.append(_site_id)
				# sql = "INSERT INTO add_user( password, mobileno,site_id) VALUES(%s,%s, %s)"
				# data = ( _password,_mobileno,_site_id,)
				# conn = mysql.connect()
				# cursor = conn.cursor()
				# cursor.execute(sql, data)
				# conn.commit()
				# cursor = conn.cursor(pymysql.cursors.DictCursor)

				data_user["mobileno"] = mobileno_list
				data_user["password"] = password_list
				data_user["site_id"] = site_id_list
				data.to_csv('hurling_out_extended.csv') 

				msg=_json

				msg['success']='true'

				msg['message']='user registration successfull.'


				msg['token'] = jwt.encode(
					{"user_id": _json['mobile_no']},
					SECRET_KEY,
					algorithm="HS256"
				)



				resp=jsonify(msg)

				print('response',resp)
				return resp
				# if cursor.execute("SELECT * FROM userlogin WHERE type=%s",(_type,)):
				#   account = cursor.fetchone()
				#   resp=jsonify("user Added Successfully")
				#   return resp

				# # sql = "SELECT * FROM userlogin WHERE email=%s AND password=%s",(_email, _password,)
				# # resp = cursor.fetchone()
				# return resp
		else:
			return not_found()
	except Exception as e:
		print("exception",e)
	finally:
		cursor.close() 
		conn.close()



@app.route('/updateuser', methods=['POST'])
@cross_origin()
def update_userupdateuser():
	conn = None
	cursor = None
	try:
		_json=request.json
		print(request.json)
		_password=_json['password']
		_mobileno=_json['mobileno']
		_name=_json['name']
		# # validate the received values
		# if _email and _password and request.method == 'POST':
		#     conn = mysql.connect()

		#     cursor = conn.cursor(pymysql.cursors.DictCursor)
		#     if cursor.execute("SELECT * FROM adduser WHERE email=%s AND password=%s",(_email, _password,)):
		#         resp = jsonify("Pls do not enter duplicate Email Id And Password")
		#         return resp
		#     elif cursor.execute("SELECT * FROM adduser WHERE aadhar_number=%s",(_aadharnumber,)):
		#         resp = jsonify("Pls do not enter duplicate Aadhar Number")
		#         return resp 
		#     else:
		#         sql = "INSERT INTO adduser(aadhar_number, aadhar_image, password, mobileno, email, status, activeemail, type) VALUES(%s, %s, %s, %s, %s,%s,%s,%s)"
		#         data = (_aadharnumber, _aadharimage, _password,_mobileno,_email,_status,_activeemail,_type,)
		#         conn = mysql.connect()
		# _json = request.json
		# _cd = _json['countryid']
		# _stateid=_json['stateid']
		# _citiid=_json['citiid']
		# _areaname = _json['areaname']
		# _areaid=_json['areaid']
		if _id and request.method == 'POST':
			sql = "UPDATE add_user SET password=%s,mobileno=%s,name=%s WHERE id=%s"
			data = (_password,_mobileno,_name,_id,)
			conn = mysql.connect()
			cursor = conn.cursor()
			cursor.execute(sql,data)
			conn.commit()
			resp = jsonify('Record updated successfully!')
			resp.status_code = 200
			return resp
		else:
			return not_found()
	except Exception as e:
		print(e)
	finally:
		cursor.close() 
		conn.close() 














SECRET_KEY = os.environ.get('SECRET_KEY') or 'this is a secret'

@app.route('/ear_images', methods=['GET','POST'])
def ear_images():

	path = os.getcwd()
	print(path)


	if request.method == "POST":
		# try:
		print("request ",request.form)
		_form = request.form

		_form=dict(_form)
		print("request ",_form)
		_name = _form['name']
		_token = _form['token']
		_site_id = _form['site_id']
		_father_name = _form['fname']
		_dob = _form['dob']
		_files = request.files
		print("_files",_files)
		byte_video=_files['video'].read()

		Time_s=str(datetime.datetime.now())
		Time_s=Time_s.split(' ')[1]

		Time_s=Time_s.split('.')[0]
		Time_s=Time_s.split(":")

		S_time=Time_s[0]+'_'+Time_s[1]


		with open('video_register'+'_'+S_time+'.mp4', 'wb') as out:
			out.write(byte_video)

		unique_id=_name+"_"+_father_name+"_"+_dob

		# conn = mysql.connect()
		# cursor = conn.cursor(pymysql.cursors.DictCursor)
		# sql = "INSERT INTO user_details( name,token,site_id,father_name,dob,unique_id) VALUES(%s,%s, %s, %s, %s, %s)"
		# data = ( _name,_token,_site_id,_father_name,_dob,_unique_id)
		# conn = mysql.connect()
		# cursor = conn.cursor()
		# cursor.execute(sql, data)
		# conn.commit()
		# cursor = conn.cursor(pymysql.cursors.DictCursor)


		try:

			cap=cv2.VideoCapture('video_register'+'_'+S_time+'.mp4')
			

			save_path_site_id_ear = os.path.join(path,'images',_site_id+"_ear")

			if os.path.exists(save_path_site_id_ear)==False:
				os.mkdir(save_path_site_id_ear)

			# save_path_face = os.path.join(save_path_site_id_face ,unique_id)

			# if os.path.exists(save_path_face) == False:
			# 	os.mkdir(save_path_face)

			


			save_path_ear = os.path.join(save_path_site_id_ear ,unique_id)

			if os.path.exists(save_path_ear) == False:
				os.mkdir(save_path_ear)
				count=1
			else:
				images_existing=os.listdir(save_path_ear)
				images_existing.sort(key=lambda f:int(re.sub('\D','',f)))
				count=int(images_existing[-1][:-4])+1

			while True:

				s,frame=cap.read()

				if frame is None:
					cap.release()
					break




				if count%2==0:

					try:





						ear_results = ear_detection_model.predict(frame,device="cuda:1",classes=0)

						for box in ear_results[0].boxes:

							#id_tracker=tracker_ids[II]
							xmin, ymin, xmax, ymax = box.xyxy[0].cpu().detach().numpy()

							if xmin<0:
								xmin=0
							if ymin<0:
								ymin=0

							# X.append(xmin)
							# X.append(xmax)
							# Y.append(ymin)
							# Y.append(ymax)
							# boxes.append({"x1":xmin,"y1":ymin,"x2":xmax,"y2":ymax})
							cropped_frame_ear = frame[int(ymin):int(ymax), int(xmin):int(xmax)]



						







						file_name=str(count)+".png"

						cv2.imwrite(save_path_ear+'/'+file_name,cropped_frame_ear)
					except Exception as e:
						print(e)


				count=count+1




			msg={"success":"true","message":"Ear registration successfull."}

			resp = jsonify(msg)
			print('response',resp)
			return resp

		except Exception as e:

			print("error",e)

			msg={"success":"false","message":"Ear registration failed."}

			resp = jsonify(msg)
			print('response',resp)
			return resp

# finally:
# 	cursor.close() 
# 	conn.close()		






@app.route('/login', methods=['POST','GET'])
@cross_origin()
def add_login1():
	conn = None
	cursor = None
	# msg = ""
	# session=None

	try:
		print("request ",request.json)
		_json = request.json
		print("request ",_json)
		# _name = _json['name']
		_mobileno = _json['mobile_no']
		_password = _json['password']
		_DeActive="DeActive"
		# _type= _json['type']
		# validate the received values

		data_user=pd.read_csv("user.csv")

		mobileno_list=data_user["mobileno"]
		password_list=data_user["password"]
		site_id_list=data_user["site_id"]
		if _mobileno and _password and request.method == 'POST':
			conn = mysql.connect()
			cursor = conn.cursor(pymysql.cursors.DictCursor)
			if _mobileno not in mobileno_list:

				msg={"success":"false","message":"user login failed due to wrong mobileno."}
				resp=jsonify(msg)
				print('wrong email',resp)
				return resp
			elif _password not in password_list:
				msg={"success":"false","message":"user login failed due to wrong password."}
				resp=jsonify(msg)
				print('wrong email',resp)
				return resp    
			elif _mobileno in mobileno_list and _password in password_list:

				for mobile_n ,pass_n,site_id_n in zip(mobileno_list, password_list,site_id_list):
					if mobile_n==_mobileno and pass_n==_password:
						site_id_nnn=site_id_n


				rows={"mobileno":_mobileno,"password":_password,"site_id":site_id_nnn}

				# rows = cursor.fetchone()
				print(rows)
				
				rows['token'] = jwt.encode(
					{"user_id": _json['mobile_no']},
					SECRET_KEY,
					algorithm="HS256"
				)

				rows['success']='true'
				rows['message']='user login successfull.'
				resp = jsonify(rows)
				print('response',resp)
				return resp     
		else:
			return not_found()
	except Exception as e:
		print('error',e)
	finally:
		cursor.close() 
		conn.close()




#ear_net_val_24july_opt.pt

backbone_face = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x




backbone_face.head = Identity()

input_size=[384,384]
data_root='data/faces_aligned'
# model_root="ear_val_4dec.pt"
embedding_size=2048
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


checkpoint = torch.load( "face_net_val_2aug_opt.pt") #caformer_ear_net_val_noweb_27oct_opt.pt
backbone_face.load_state_dict(checkpoint['model_state_dict'])
# backbone.load_state_dict(torch.load(model_root, map_location=torch.device("cpu")))
backbone_face.to(device)
backbone_face.eval()




backbone_ear = timm.create_model('caformer_s36.sail_in22k_ft_in1k_384', pretrained=True)
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x




backbone_ear.head.fc.fc2 = Identity()

input_size=[384,384]
# data_root='data/faces_aligned'
# model_root="ear_val_4dec.pt"
embedding_size=2048
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


checkpoint = torch.load( "caformer_ear_net_val_noweb_27oct_opt.pt") #caformer_ear_net_val_noweb_27oct_opt.pt #caformer_ear_net_avg_27aug_opt_best_latest.pt
backbone_ear.load_state_dict(checkpoint['model_state_dict'])
# backbone.load_state_dict(torch.load(model_root, map_location=torch.device("cpu")))
backbone_ear.to(device)
backbone_ear.eval()
	









@app.route('/get_embeddings', methods=['POST',"GET"])
@cross_origin()
def get_embeddings( ):
	_json = request.json
	_site_id_ear = _json['site_id']+"_ear"
	embedding_size=2048

	data_root='images/'+_site_id_ear+'/'


	Total_images=0
	names=os.listdir(data_root)

	for name in names:
		images=os.listdir(data_root+'/'+name)
		Total_images=len(images)+Total_images


	embedding_size=2048

	


	Total_images=0
	


	for name in names:
		images=os.listdir(data_root+'/'+name)
		Total_images=len(images)+Total_images

	embeddings_ear = np.zeros([Total_images, 2048])

	device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


	print(names)
	NAMES=[]

	IMAGES=[]
	i=-1

	for name in names:
		images=os.listdir(data_root+'/'+name)

		for img in images:
			i+=1
			print(data_root+'/'+name+'/'+img)
		

			image11=cv2.imread(data_root+'/'+name+'/'+img)
			image11=cv2.resize(image11,(384,384))
			image1=cv2.cvtColor(image11,cv2.COLOR_BGR2RGB)
			image1 = image1.transpose(2,0,1)

			
			image1=image1/255.0
			

			# image1=Image.fromarray(image1) 
			image1=[image1]          

			frame=np.array(image1).reshape(1,3,384,384)

			print(frame.shape)

			frame=torch.from_numpy(frame).float()

			print(f"Number of classes: {len(names)}")
			

			
			with torch.no_grad():
				print(i)
				embeddings_ear[i, :] = F.normalize(backbone_ear(frame.to(device))).cpu()

				IMAGES.append(image11)
				NAMES.append(name)
	with open('embeddings_384'+'_'+_site_id_ear+'.pickle','wb') as f:
		pickle.dump(embeddings_ear,f)

	with open('names_384'+'_'+_site_id_ear+'.pickle','wb') as f1:
		pickle.dump(NAMES,f1)
			
	msg={"success":True,"message":"embeddings saved"}

	resp = jsonify(msg)
	print('response',resp)
	return resp






@app.route('/attendance', methods=['POST'])
@cross_origin()
def get_frames3():

	_file = request.files
	byte_video=_file['video'].read()

	Time_s=str(datetime.datetime.now())
	Time_s=Time_s.split(' ')[1]

	Time_s=Time_s.split('.')[0]
	Time_s=Time_s.split(":")

	S_time=Time_s[0]+'_'+Time_s[1]


	with open('video_test'+'_'+S_time+'.mp4', 'wb') as out:
		out.write(byte_video)

	# result_crop_face = cv2.VideoWriter('video_test_face'+'_'+S_time+'_cropped_384.mp4',
	#                          cv2.VideoWriter_fourcc(*'MJPG'),
	#                          10, (384,384))

	result_crop_ear = cv2.VideoWriter('video_test_ear'+'_'+S_time+'_cropped_384.mp4',
	                         cv2.VideoWriter_fourcc(*'MJPG'),
	                         10, (384,384))
	_form=request.form

	_site_id_face=_form['site_id'] +"_face"

	_site_id_ear=_form['site_id'] +"_ear"

	embeddings_test_face = np.zeros([1, embedding_size])
	embeddings_test_ear = np.zeros([1, 2048])
	# with open('embeddings_384'+'_'+_site_id_face+'.pickle','rb') as f:
	# 	embeddings_saved_face=pickle.load(f)

	# with open('names_384'+'_'+_site_id_face+'.pickle','rb') as f1:
	# 	NAMES_saved_face=pickle.load(f1)


	with open('embeddings_384'+'_'+_site_id_ear+'.pickle','rb') as f:
		embeddings_saved_ear=pickle.load(f)

	with open('names_384'+'_'+_site_id_ear+'.pickle','rb') as f1:
		NAMES_saved_ear=pickle.load(f1)


	# print('please Enter name of video file')




	cap=cv2.VideoCapture('video_test'+'_'+S_time+'.mp4')

	msg={
	    "success":False,
	    "message":"Not match.",
	}
	names_freq_ear=[]
	names_freq_face=[]
	max_sim={}

	count_frame=0


	while True:
		s,frame=cap.read()
		# print(image1.shape)

		try:

			if frame is None:
				break


			# image1=cv2.imread('IMG_20191008_172539_287.jpg')
			
			(h, w) = frame.shape[:2]



			# if True:

			# 	try:




			
			
			# 		imgg1=frame.copy()


			# 		detections = detector.detect(imgg1)


			# 		for det__ in detections:
						
			# 			startX_z, startY_z,endX_z, endY_z,confidence_z=det__
			# 			if confidence_z > 0.4:
			# 				startX_z, startY_z,endX_z, endY_z=int(startX_z)-30, int(startY_z)-70,int(endX_z)+20, int(endY_z)+30

			# 				y_z = startY_z - 10 if startY_z - 10 > 10 else startY_z + 10
			# 				image=imgg1[startY_z: endY_z,startX_z:endX_z]


			# 		image=cv2.resize(image,(384,384))

			# 		result_crop_face.write(image)

			# 		image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

			# 		image = image.transpose(2,0,1)
			# 		image=image/255.0
					

			# 		# image1=Image.fromarray(image1) 
			# 		image=[image]          

			# 		frame_=np.array(image).reshape(1,3,384,384)
			# 		frame_=torch.from_numpy(frame_)
			# 		# frame=image.resize(1,3,112,112)
			# 		# backbone = Backbone(input_size)
			# 		# backbone.load_state_dict(torch.load(model_root, map_location=torch.device("cpu")))
			# 		# backbone.to(device)
			# 		# backbone.eval()
			# 		embeddings_test_face[0, :] = F.normalize(backbone_face(frame_.to(torch.device("cuda:1" if torch.cuda.is_available() else "cpu")).float())).detach().cpu().numpy()




			# 		cos_similarity_face = np.dot(embeddings_saved_face, embeddings_test_face.T)




			# 		cos_similarity_face = cos_similarity_face.clip(min=0, max=1)
			# 		print('similarity score_face',max(cos_similarity_face))
			# 		if max(cos_similarity_face)>0.5:
			# 			idx=np.argmax(cos_similarity_face)
			# 			max_sim[idx]=np.max(cos_similarity_face)

			# 			# cv2.rectangle(imgg1, (x_min_ear, y_min_ear), (x_max_ear, y_max_ear),
			# 			# 	(0, 0, 255), 2)
			# 			# cv2.putText(imgg1, NAMES_saved[idx], (x_min_ear, y_min_ear),
			# 			# 	cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

			# 			# conn = mysql.connect()
						
			# 			# cursor = conn.cursor(pymysql.cursors.DictCursor)
			# 			#print('inserting data')

			# 			names_freq_face.append(idx)

			# 		if max(cos_similarity_face)<=0.5:
			# 			idx=np.argmax(cos_similarity_face)

			# 			# cv2.rectangle(imgg1, (x_min_ear, y_min_ear), (x_max_ear, y_max_ear),
			# 			# 	(0, 0, 255), 2)
			# 			# cv2.putText(imgg1, 'unknown', (x_min_ear, y_min_ear),
			# 			# 	cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)




			# 		# image1[startY-80: endY+80,startX-100:endX+100]=imgg1

			# 	except:
			# 		cc=1

			if True:


				ear_results = ear_detection_model.predict(frame,device="cuda:1",classes=0)

				for box in ear_results[0].boxes:

					#id_tracker=tracker_ids[II]
					xmin, ymin, xmax, ymax = box.xyxy[0].cpu().detach().numpy()

					if xmin<0:
						xmin=0
					if ymin<0:
						ymin=0

					# X.append(xmin)
					# X.append(xmax)
					# Y.append(ymin)
					# Y.append(ymax)
					# boxes.append({"x1":xmin,"y1":ymin,"x2":xmax,"y2":ymax})
					image = frame[int(ymin):int(ymax), int(xmin):int(xmax)]



				image=cv2.resize(image,(384,384))

				result_crop_ear.write(image)

				image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

				image = image.transpose(2,0,1)
				image=image/255.0
				

				# image1=Image.fromarray(image1) 
				image=[image]          

				frame_=np.array(image).reshape(1,3,384,384)
				frame_=torch.from_numpy(frame_)
				# frame=image.resize(1,3,112,112)
				# backbone = Backbone(input_size)
				# backbone.load_state_dict(torch.load(model_root, map_location=torch.device("cpu")))
				# backbone.to(device)
				# backbone.eval()
				embeddings_test_ear[0, :] = F.normalize(backbone_ear(frame_.to(torch.device("cuda:1" if torch.cuda.is_available() else "cpu")).float())).detach().cpu().numpy()




				cos_similarity_ear = np.dot(embeddings_saved_ear, embeddings_test_ear.T)




				cos_similarity_ear = cos_similarity_ear.clip(min=0, max=1)
				print('similarity score ear',max(cos_similarity_ear))
				if max(cos_similarity_ear)>0.75:
					idx=np.argmax(cos_similarity_ear)
					max_sim[idx]=np.max(cos_similarity_ear)

					# cv2.rectangle(imgg1, (x_min_ear, y_min_ear), (x_max_ear, y_max_ear),
					# 	(0, 0, 255), 2)
					# cv2.putText(imgg1, NAMES_saved[idx], (x_min_ear, y_min_ear),
					# 	cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)

					# conn = mysql.connect()
					
					# cursor = conn.cursor(pymysql.cursors.DictCursor)
					#print('inserting data')

					names_freq_ear.append(idx)

				if max(cos_similarity_ear)<=0.75:
					idx=np.argmax(cos_similarity_ear)

					# cv2.rectangle(imgg1, (x_min_ear, y_min_ear), (x_max_ear, y_max_ear),
					# 	(0, 0, 255), 2)
					# cv2.putText(imgg1, 'unknown', (x_min_ear, y_min_ear),
					# 	cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4)




				# image1[startY-80: endY+80,startX-100:endX+100]=imgg




		except Exception as e:
			print(e)

	# if len(names_freq_ear)>10:


	# 	frequency = collections.Counter(names_freq_ear)

	# 	frequency=dict(frequency)


	# 	try:

	# 		pair_ear=max(frequency.items(), key=lambda k: k[1])

	# 		# total = sum(item.get(pair[0],0) for item in max_sim) / len(max_sim)
	# 	except Exception as e:
	# 		print(e)


	if len(names_freq_ear)>10:


		frequency = collections.Counter(names_freq_ear)

		frequency=dict(frequency)


		try:

			pair_ear=max(frequency.items(), key=lambda k: k[1])

			# total = sum(item.get(pair[0],0) for item in max_sim) / len(max_sim)
		except Exception as e:
			print(e)



		result_crop_ear.release()
		# result_crop_face.release()

		Time=str(datetime.datetime.now())
		Time=Time.split(' ')[1]
		today = date.today()
		Date = today.strftime("%m/%d/%y")

		# sql = "INSERT INTO present_employees(Name,SITE_ID, Time, Date) VALUES(%s,%s,%s,%s)"
		# data = (NAMES_saved_ear[pair_ear[0]],_site_id_ear,Time,Date)
		# conn = mysql.connect()
		# cursor = conn.cursor()
		# cursor.execute(sql, data)
		# conn.commit()
		msg={"success":True,
				"message":"attendance marked successfully.",
			    "name":str(NAMES_saved_ear[pair_ear[0]]),
			    "fname":'1234',
			    "site_id":str(_site_id_ear),
			    "dob":"11-01-2010"
			}

	else:

		msg={"success":False,
				"message":"Not matched."
			}

	resp = jsonify(msg)
	print('response',resp)
	return resp 			




if __name__=="__main__":
	# app.run(debug=True)
	app.run(debug=True,port=80,host='0.0.0.0')