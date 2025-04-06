import glob
import numpy as np
import time
import cv2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from utilities.customUtils import *
from dataTools.dataNormalization import *
from dataTools.customTransform import *
import os
class customDatasetReader(Dataset):
    def __init__(self, image_list, imagePathGT, height, width, transformation=True):
        self.image_list = image_list
        self.imagePathGT = imagePathGT
        self.transformLR = transforms
        self.imageH = height
        self.imageW = width
        normalize = transforms.Normalize(normMean, normStd)

        self.transformHRGT = transforms.Compose([ transforms.Resize((self.imageH, self.imageW)),
                                                transforms.Resize((self.imageH,self.imageW), interpolation=Image.BICUBIC),
                                                transforms.ToTensor(),
                                                normalize,
                                                ])

    
        self.transformRI = transforms.Compose([ transforms.Resize((self.imageH, self.imageW)),
                                                transforms.ColorJitter(brightness=(0.1,0.8), contrast=(0.1,0.8)),
                                                transforms.ToTensor(),
                                                normalize,
                                                AddGaussianNoise(pov=1.5)
                                            ])

    def __len__(self):
        return (len(self.image_list))
    
    def __getitem__(self, i):

        # Read Images
        #print ("print i",i, i+1)
        try:    
            self.sampledImage = Image.open(self.image_list[i])
        except:
            self.sampledImage = Image.open(self.image_list[i + 1])
            os.remove(i)
            print("File deleted:", i)
            i += 1

        # Convert sampledImage to RGB if it is grayscale
        if self.sampledImage.mode != 'RGB':
            self.sampledImage = self.sampledImage.convert('RGB')

        self.gtImageFileName = self.imagePathGT + extractFileName(self.image_list[i])
        self.gtImage = Image.open(self.gtImageFileName)

        # Convert gtImage to RGB if it is grayscale
        if self.gtImage.mode != 'RGB':
            self.gtImage = self.gtImage.convert('RGB')

        # Transforms Images for training 
        self.inputImage = self.transformRI(self.sampledImage)
        self.gtImageHR = self.transformHRGT(self.gtImage)

        #print (self.gtImageHR.max(), self.gtImageHR.min(), self.inputImage.max(), self.inputImage.min())


        return self.inputImage, self.gtImageHR
