import pycocotools
import torch
import random
from torchvision.transforms import Compose,ToTensor,CenterCrop,Grayscale,ToPILImage
from pycocotools.coco import COCO
import PIL.Image as Image
import glob
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MetaCOCOMaskDataset(torch.utils.data.Dataset):
    def __init__(self,trainfolder,valfolder,trainjson,valjson,n_way,k_shot,size=(64,64),transform=None):
        super(MetaCOCOMaskDataset,self).__init__()
        self.traincoco=COCO(trainjson)
        self.valcoco=COCO(valjson)
        cats = self.traincoco.loadCats(self.traincoco.getCatIds())
        self.classes=[cat['id'] for cat in cats]
        self.n_way=n_way
        self.k_shot=k_shot
        #self.transform=lambda inputsize,norm=True:transform if transform else Compose([lambda x:ToTensor()(Grayscale()(x)) if norm else lambda x:torch.from_numpy(x).unsqueeze(0) ,lambda x:F.interpolate(x.unsqueeze(0),inputsize)])

        self.trainfolder=trainfolder
        self.valfolder=valfolder
        self.size=size
    def transform(self,x,norm=True):
        if norm:
            x=Grayscale()(x)
            x=ToTensor()(x)
        else:
            x=torch.from_numpy(x)
            x=x.type(torch.float32)
            x=x.unsqueeze(0)
        x = x.unsqueeze(0)
        x=F.interpolate(x,self.size)
        return x
    def __len__(self):
        return len(self.classes)
    def __getitem__(self, item):
        train_raw=[]
        train_mask=[]
        val_raw=[]
        val_mask=[]
        #select random class (N - way)
        sampled_classes=random.sample(self.classes,self.n_way)
        #select random file (K - shot)
        for rawl,maskl,COCO,folder in [[train_raw,train_mask,self.traincoco,self.trainfolder],[val_raw,val_mask,self.valcoco,self.valfolder]]:
            for cls in sampled_classes:
                ids=[COCO.loadImgs(it) for it in  random.sample(COCO.getImgIds(catIds=cls),self.k_shot)]
                rawl+=[self.transform(Image.open(f'{folder}/{id[0]["file_name"]}')) for id in ids]
                maskl+=[self.transform(COCO.annToMask(COCO.loadAnns(COCO.getAnnIds(imgIds=id[0]['id'],catIds=cls,iscrowd=None))[0]).astype(np.float32),False) for id in ids]
        #return train[raw_img(N*K,C,H,W), mask_img(N*K,C,H,W)], test[raw_img(N*K,C,H,W),mask_img(N*K,C,H,W)]
        return {'train':(torch.cat(train_raw),torch.cat(train_mask)),'test':(torch.cat(val_raw),torch.cat(val_mask))}

class MetaOWNCrackDataset(torch.utils.data.Dataset):
    def __init__(self,folder,k_shot,size=(256,256),transform=None,batchsize=8):
        super(MetaOWNCrackDataset,self).__init__()
        self.folder=folder
        self.k_shot=k_shot
        self.transform=transform if transform else Compose([Grayscale(),ToTensor(),lambda x:F.interpolate(x.unsqueeze(0),self.size).squeeze(0)])
        self.trainimg=[]
        self.imgs=glob.glob(f'{self.folder}/raw/*.jpg')
        self.size = size
        self.tomask=lambda x:x.replace('raw','mask').replace('jpg','png')
        self.batchsize=batchsize
        self.trainimg = random.sample(self.imgs, self.k_shot)
        self.testimg=list(set(self.imgs)-set(self.trainimg))
    def __len__(self):
        return len(self.testimg)

    def __getitem__(self, item):
        imgp=self.testimg[item]
        return self.transform(Image.open(imgp)), self.transform(Image.open(self.tomask(imgp)))

    def train(self):
        pass
        #one class (1 - way)
        #select random file (K- shot)

        raw=torch.stack([self.transform(Image.open(imgp)) for imgp in self.trainimg])
        mask=torch.stack([self.transform(Image.open(self.tomask(imgp))) for imgp in self.trainimg])
        #return train[raw(K,C,H,W), mask(K,C,H,W)], test[raw(K,C,H,W), mask(K,C,H,W)]
        return {'train':(raw,mask)}

