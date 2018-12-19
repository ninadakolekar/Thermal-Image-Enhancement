#Usage: 

import torch

from dataset import dataset
import network
import imageio

import os
import sys
import logging
import random
from time import time

test_dir = sys.argv[1]      
output_dir = sys.argv[2]
input_mode = sys.argv[3]

if input_mode == None:
	input_mode = 'rgb'

elif input_mode not in ['rgb','gs']:
	print("Unrecognized input mode specified")
	exit()

#Logging config

logging.basicConfig(filename=os.path.join(output_dir,'test_log.txt'),filemode='w',format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',datefmt='%H:%M:%S',level=logging.INFO)

# Global Vars

batch_size = 128
cuda = torch.cuda.is_available()

if cuda:
	torch.cuda.manual_seed(2)

# Loss function

def compute_loss(pred,gt):
	criterion = torch.nn.MSELoss()
	return criterion(pred,gt)

# Test batch generator

def test_dataset(test_dir,batch_size):
	assert(os.path.exists(test_dir))
	
	images = []
	for in_file in os.listdir(test_dir):
		if isImageFile(in_file):
			img_path = os.path.join(test_dir,in_file)
			if input_mode=='gs'
				img = (imageio.imread(img_path))
			elif input_mode=='rgb':
				img = (imageio.imread(img_path)).dot([0.299, 0.587, 0.114])
			img = from_numpy(img).unsqueeze(0).unsqueeze(0).float()
			images.append(img)
	
	l = len(images)
	for idx in range(0,l,batch_size):
		yield stack(images[idx:min(idx+batch_size,l)])

# Test function
def test(test_dir,out_dir,verbose=False):

	model = network.ten()

	logging.info('TEN Model loaded')
	if verbose:
		print('TEN Model loaded')
		total_params = sum(p.numel() for p in model.parameters())
		print(f'Total Parameters: {total_params}')

	if cuda:
		model = model.cuda()i

	model.eval()
	
	
	
