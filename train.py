# Usage: python train.py ../dataset/rgb91 2 100 output/

import torch

from dataset import dataset
import network
from utils import blur,upscale,save_checkpoint
import imageio

import os
import sys
import logging
import random
from time import time

dataset_path = sys.argv[1]      
scale_factor = int(sys.argv[2])
num_epochs = int(sys.argv[3])
output_dir = sys.argv[4]

# Logging config

logging.basicConfig(filename=os.path.join(output_dir,'log.txt'),filemode='w',format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',datefmt='%H:%M:%S',level=logging.INFO)

# Global Vars and Hyperparameters

learning_rate = 0.001
weight_decay = 5e-4
batch_size = 128
cuda = torch.cuda.is_available()

if cuda:
	torch.cuda.manual_seed(2)

# Loss function

def compute_loss(pred,gt):
	criterion = torch.nn.MSELoss()
	return criterion(pred,gt)

# Training and validation function

def train_and_validate(dataset_path,batch_size,scale_factor,num_epochs,learning_rate,weight_decay,output_dir,verbose=True):

	model_output_dir = os.path.join(output_dir,'model')

	model = network.ten()
	
	logging.info('TEN Model loaded')
	if verbose:
		print('TEN Model loaded')
		total_params = sum(p.numel() for p in model.parameters())
		print(f'Total Parameters: {total_params}')

	if cuda:
		model = model.cuda()
	
	model.train()
	
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)

	logging.info('Adam optimizer loaded')
	if verbose:
		print('Adam optimizer loaded')

	optimizer.zero_grad()

	total_epoch_time = 0
	losses = []
	
	# Set initial best_loss arbitrarily high
	best_val_loss = 2.0e50
	
	for epoch in range(1,num_epochs+1):

		model.train()
	
		logging.info(f'Epoch {epoch} Start')
		if verbose:
			print(f'\n<----- START EPOCH {epoch} ------->\n')

		start_time = time()
		
		total_loss_for_this_epoch = 0
		
		# For each batch
		batch = 1
		for patches,gt in dataset(dataset_path,batch_size,scale_factor):

			if cuda:
				patches = patches.cuda()
				gt = gt.cuda()

			pred = model(patches)
			loss = compute_loss(pred,gt)
			
			logging.info(f'Epoch {epoch} Batch {batch} Loss {loss.item()}')
			if verbose and (batch-1)%10==0:
				print(f'Epoch {epoch} Batch {batch} Loss {loss.item()}')
			
			loss.backward()
			optimizer.step()
			
			total_loss_for_this_epoch += loss.item()

			batch+=1

		avg_loss = total_loss_for_this_epoch/batch
		losses.append(avg_loss)

		epoch_time = time()-start_time
		if verbose:
			print(f'Epoch time: {epoch_time}')
		total_epoch_time += epoch_time

		# Validation
		model.eval()
		val_img_file = random.choice([f for f in os.listdir(dataset_path) if f.endswith('.bmp')])
		val_img = imageio.imread(os.path.join(dataset_path,val_img_file)).dot([0.299, 0.587, 0.114])
		mod_val_img = torch.from_numpy(blur(upscale(val_img,scale_factor),scale_factor)).float().unsqueeze(0).unsqueeze(0)
		val_img = torch.from_numpy(upscale(val_img,scale_factor)).float().unsqueeze(0).unsqueeze(0)
		if cuda:
			mod_val_img = mod_val_img.cuda()
			val_img = val_img.cuda()
		out = model(mod_val_img)
		val_loss = compute_loss(out,val_img).item()
		
		if verbose:
			print(f'Epoch {epoch} Validation Image {val_img_file} Loss {val_loss}')
		
		# Save current model
		save_checkpoint({'epoch':epoch,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),},model_output_dir,'current.pth')
		logging.info('Current model saved')
		if verbose:
                        print('Current model saved')

		# Save best model
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			save_checkpoint({'epoch':epoch,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),},model_output_dir,'best.pth')	
			logging.info('Best model saved')
			if verbose:
		        	print('Best model saved')

		# Save model every 20 epochs
		if (epoch)%20 == 0:
			save_checkpoint({'epoch':epoch,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),},model_output_dir,f'epoch_{epoch}.pth')
			logging.info(f'Epoch {epoch} Model saved')
			if verbose:
                        	print(f'Epoch {epoch} Model saved')

		# Learning rate decay

		if epoch%30 == 0 and epoch <=60:
			learning_rate = learning_rate/10
			for param_group in optimizer.param_groups:
				param_group['lr'] = learning_rate
			logging.info(f'Epoch {epoch}: Learning rate decayed by factor of 10')

	
		logging.info(f'Epoch {epoch} completed')
		if verbose:
			print(f'\n<----- END EPOCH {epoch} Time elapsed: {time()-start_time}------->\n')	

	logging.info('All epochs completed')
	logging.info(f'Average Time: {total_epoch_time/num_epochs:.4f} seconds')
	logging.info(f'Average Loss: {sum(losses) / len(losses):.4f}')
	if verbose:
		print('All epochs completed')
		print(f'Average Time: {total_epoch_time/num_epochs:.4f} seconds')
		print(f'Average Loss: {sum(losses) / len(losses):.4f}')

	if verbose:
		print('Losses array: ',losses)
		print('Best Validation Loss',best_val_loss)

train_and_validate(dataset_path,batch_size,scale_factor,num_epochs,learning_rate,weight_decay,output_dir,True)


