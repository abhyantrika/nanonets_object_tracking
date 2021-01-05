import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from siamese_dataloader import *
from siamese_net import *
from scipy.stats import multivariate_normal
from tqdm import tqdm


class Config():
	training_dir = "/media/ADAS1/MARS/bbox_train/bbox_train/"
	testing_dir = "/media/ADAS1/MARS/dummy_test_set/bbox_test/"

transforms = torchvision.transforms.Compose([
	torchvision.transforms.Resize((128,128)),
	torchvision.transforms.ToTensor()
	])


def get_gaussian_mask():
	#128 is image size
	x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]
	xy = np.column_stack([x.flat, y.flat])
	mu = np.array([0.5,0.5])
	sigma = np.array([0.22,0.22])
	covariance = np.diag(sigma**2) 
	z = multivariate_normal.pdf(xy, mean=mu, cov=covariance) 
	z = z.reshape(x.shape) 

	z = z / z.max()
	z  = z.astype(np.float32)

	mask = torch.from_numpy(z)

	return mask


class Siamese_Triplet_Test(Dataset):
	
	def __init__(self,imageFolderDataset,transform=None,should_invert=True):
		self.imageFolderDataset = imageFolderDataset    
		self.transform = transform
		self.should_invert = should_invert
		
	def __getitem__(self,index):
		# Get a random image which will be used as an anchor
		img0_tuple = random.choice(self.imageFolderDataset.imgs)
		# img0_tuple = (img_path, class_id)

		negative_images = set()
		while len(negative_images) < 32:
			# keep looping till a different class image is found [Negative image]
			img1_tuple = random.choice(self.imageFolderDataset.imgs)

			if img0_tuple[1] == img1_tuple[1]:
				# Comparing class_id, if same as anchor image, this won't use used as a positive image
				continue
			else:
				# If class id is different than anchor image, this will be used as a negative image
				negative_images.update([img1_tuple[0]])	# Keeping the image_path of negative images

		negative_images = list(negative_images)

		# Getting anchor image and class name
		anchor_image_name = img0_tuple[0].split('/')[-1]
		anchor_class_name = img0_tuple[0].split('/')[-2]

		# Getting all the images which belong to the same class as anchor image.
		all_files_in_class = glob.glob(self.imageFolderDataset.root+anchor_class_name+'/*')
		# Only those images which belong to the same class as anchor image but isn't anchor image will 
		# be selected as a candidate for positive sample
		all_files_in_class = [x for x in all_files_in_class if x!=img0_tuple[0]]
			
		if len(all_files_in_class)==0:
			# If there is no image (other than anchor image) belonging to the anchor image class, anchor 
			# image will be taken as positive sample
			positive_image = img0_tuple[0]
		else:
			# Choose random image (of same class as anchor image) as positive sample
			positive_image = random.choice(all_files_in_class)

		if anchor_class_name != positive_image.split('/')[-2]:
			print("Error") # Checking if the class of both anchor and positive image is same


		anchor = Image.open(img0_tuple[0])
		#negative = Image.open(img1_tuple[0])
		positive = Image.open(positive_image)

		anchor = anchor.convert("RGB")
		#negative = negative.convert("RGB")
		positive = positive.convert("RGB")
		
		if self.should_invert:
			anchor = PIL.ImageOps.invert(anchor)
			positive = PIL.ImageOps.invert(positive)
			#negative = PIL.ImageOps.invert(negative)

		if self.transform is not None:
			anchor = self.transform(anchor)
			positive = self.transform(positive)
			#negative = self.transform(negative)

		# Multiple negative samples for each anchor-positive pair will be parsed
		negs = []
		for i in range(len(negative_images)):
			neg_image = Image.open(negative_images[i])
			if self.should_invert:				
				neg_image = PIL.ImageOps.invert(neg_image)

			if self.transform is not None:
				neg_image = self.transform(neg_image)	
			negs.append(neg_image)

		negatives = torch.squeeze(torch.stack(negs))

		return anchor, positive, negatives

	def __len__(self):
		return len(self.imageFolderDataset.imgs)



folder_dataset_test = dset.ImageFolder(root=Config.testing_dir) # PyTorch object of image dataset dir
siamese_dataset = Siamese_Triplet_Test(imageFolderDataset=folder_dataset_test,transform=transforms,should_invert=False) # Initializing data parser class
test_dataloader = DataLoader(siamese_dataset,num_workers=12,batch_size=1,shuffle=False) # PyTorch data parser obeject
dataiter = iter(test_dataloader) # PyTorch data iterator
x0,_,_ = next(dataiter) # NOT SURE WHY THIS HAS BEEN DONE

net = torch.load('ckpts/model640.pt').cuda()
net.eval()


correct = 0
total_correct = 0

total = len(siamese_dataset)
print("Total testing images:", total)

gaussian_mask = get_gaussian_mask().cuda() # Only triplet_model 1024 has gaussian mask.

for i in tqdm(range(total-1)):
	anc,pos, negatives = next(dataiter) # Parse the data and get next sample
	# print(i)
	batch_correct = 0
	negatives = torch.squeeze(negatives)

	for j in range(len(negatives)): # Iterate over multiple negative samples
		neg = negatives[j]
		neg = neg.unsqueeze(0)

		concatenated = torch.cat((anc,pos,neg),0) # Creating a [anchor, positive, negative] sample to be passed onto the model for testing
		output1, output2, output3 = net(Variable(anc).cuda()*gaussian_mask, Variable(pos).cuda()*gaussian_mask, Variable(neg).cuda()*gaussian_mask)

		## Added to make shapes from 1024,3 to 1024x3 ##
		output1 = torch.flatten(output1)
		output2 = torch.flatten(output2)
		output3 = torch.flatten(output3)
		## Added to make shapes from 1024,3 to 1024x3 ##

		output1 = torch.unsqueeze(output1,0) #anc
		output2 = torch.unsqueeze(output2,0) #pos
		output3 = torch.unsqueeze(output3,0) #neg

		# print(output1.shape, output2.shape, output3.shape)

		d1 = F.cosine_similarity(x1=output1, x2=output2) #anc - pos
		d2 = F.cosine_similarity(x1=output1, x2=output3) #anc - neg
		# print(d1.shape, d2.shape)

		#if abs(d1 - d2) > 0.5:
		if d1 > d2: # Original Implementation
			batch_correct+=1
			#correct+=1
			total_correct+=1

	if batch_correct == len(negatives):
		correct+=1		


print('correct: ',correct)
print('completely correct batches % ',correct/(total))

print('Total correct examples: ',total_correct)
print('examplewise correct % ',total_correct/(total*len(negatives)))
