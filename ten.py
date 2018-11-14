
# coding: utf-8

# # Thermal Image Enhancement Network (TEN) - IROS 2016
# Thermal Image Enhancement using Convolution Neural Network

# In[28]:


import torch
import torchvision
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

import PIL.Image
from PIL import ImageFilter
import matplotlib
import matplotlib.pyplot as plt

import os
from time import time
from random import randint


from tqdm import tqdm


# ## Parameters

# In[29]:


batch_size = 2048
learning_rate = 0.001
num_epochs = 100
weight_decay = 5e-4
scale_factor = 2


# ## Image Transforms

# In[30]:


img_transform = transforms.Compose([
    transforms.ToTensor(),
])


# ## Dataset

# In[31]:


dataset_train_path = '/DATA1/chaitanya/ninad3/dataset/rgb91/grayscale/'
assert(os.path.exists(dataset_train_path))


# In[32]:


class rgb91(Dataset):
    
    def __init__(self,path,scale,transform=None):
        
        self.path = path
        self.scale = scale
        self.transform = transform
    
    def __len__(self):
        
        return len(os.listdir(self.path))
    
    def __getitem__(self,idx):
        
        imgList = os.listdir(self.path)
        imgName = imgList[idx]
        
        imgPath = os.path.join(self.path,imgName)
        
        img = PIL.Image.open(imgPath)
        img = img.filter(ImageFilter.GaussianBlur(radius=self.scale))
        img = img.resize((36*self.scale,36*self.scale),PIL.Image.BICUBIC)
        
        if self.transform is not None:
            img = self.transform(img)
       
        return (imgName,img)


# In[33]:


dataset = rgb91(dataset_train_path,scale_factor,transform=img_transform)
dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)


# ## Model

# In[34]:


class ten(nn.Module):
    def __init__(self):
        super(ten, self).__init__()
        self.model = nn.Sequential(

            nn.Conv2d(1,64,7,stride=1,padding=3),
            nn.ReLU(True),

            nn.Conv2d(64,32,5,stride=1,padding=2),
            nn.ReLU(True),

            nn.Conv2d(32,32,3,stride=1,padding=1),
            nn.ReLU(True),

            nn.Conv2d(32,1,3,stride=1,padding=1)
        )

    def forward(self,x):
        x = self.model(x)
        return x


# In[35]:


model = torch.nn.DataParallel(ten()).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                             weight_decay=weight_decay)


# ## Methods

# In[36]:


unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

def imsave(tensor, filepath):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
#     image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    image.save(filepath)


# In[37]:


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    learning_rate = learning_rate/10
    logs.write(f'Learning Rate set to {learning_rate}\n')
    print(f'Learning Rate set to {learning_rate}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


# In[38]:


def save_checkpoint(state, output_dir, filename):
    torch.save(state, os.path.join(output_dir,filename))


# ## Run the model

# In[ ]:


def main():

    model_output_dir = '../output/try2/model'

    logs = open('../output/try2/logs.txt',"w")
    logs.write('Thermal Enhancement Network (TEN) - Results\n\n')

    total_epoch_time = 0
    losses = []
    epoch_num = 0

    for epoch in tqdm(range(num_epochs)):

        epoch_num+=1

        logs.write('Epoch {epoch_num} Start')
        print(f'<----- START EPOCH {epoch_num} ------->\n')

        start = time()

        for ind, data in enumerate(dataloader):

            imgName, img = data
            img = Variable(img).cuda()

            # forward
            output = model(img)
            loss = criterion(output, img)

            num = randint(0,10)

            # save results
            pict = output.cpu().data
            imsave(pict[num],f'../output/try2/r_{epoch+1}_{imgName[num]}')
            logs.write(f'Input & Output Images Saved: Epoch {epoch+1} Index {ind} Image {imgName[num]}')
            print(f'Epoch {epoch+1} Index {ind} Image {imgName[num]} : Input & Output Images Saved')

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_time = time() - start
        total_epoch_time += epoch_time

        losses.append(loss.item())

        logs.write(f'Epoch {epoch+1}/{num_epochs}, loss {loss.item():.4f}, time {epoch_time:.4f} s \n')

        # Save/Update current model
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, model_output_dir, filename=f'current.pth')
        logs.write(f'Model Saved: Epoch {epoch+1} \n')

        # Save model every 20 epochs

        if epoch%20 == 0 or epoch == num_epochs-1:

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, model_output_dir, filename=f'model_{epoch+1}.pth')

            logs.write(f'Model Saved: Epoch {epoch+1} \n')
        
        # Decay learning rate by 10 every 30 epochs until 60 epochs
    
        if (epoch+1)%30 == 0 and (epoch<=59):
            learning_rate = learning_rate/10
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
            logs.write(f'Learning Rate set to {learning_rate}\n')
            print(f'Learning Rate set to {learning_rate}')

        print(f'\n<----- END EPOCH {epoch_num} ------->\n')

    print(f'Average Time: {total_epoch_time/num_epochs:.4f} seconds')
    print(f'Average Loss: {sum(losses) / len(losses):.4f}')

    logs.write(f'\n\nAverage Time: {total_epoch_time/num_epochs:.4f} seconds\n')
    logs.write(f'\nAverage Loss: {sum(losses) / len(losses):.4f}\n')
    logs.write('\n\nLOSSES Array\n[')
    for x in losses:
        logs.write(str(x))
        logs.write(', ')
    logs.write(']')
    logs.write('\n\nAll epochs completed!\n\n')
    logs.close()

if __name__ == "__main__":
    main()
