
import torch 
from torchvision import models

def get_model(name="vgg16", pretrained=True):
	if name == "resnet18":
		model = models.resnet18(pretrained=pretrained)
	elif name == "vgg16":
		model = models.vgg16(pretrained=pretrained)

		
	if torch.cuda.is_available():
		return model.cuda()
	else:
		return model 