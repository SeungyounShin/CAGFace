import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import *
from dataloader import Data_Loader
from torchvision import transforms
from optimizer import Optimizer

import matplotlib.pyplot as plt

pretrained = True
lr_start   = 1e-16 #0.0000505
visualize  = True

dl = Data_Loader('/home/yo0n/바탕화면/Flicker',256,1).loader()

if(pretrained):
    model = torch.load('./checkpoints/cand.pth')
    """
    model = nn.Sequential(
        model,
        conv3x3(3,3),
        nn.ReLU(inplace=True)
    )
    model = model.cuda()
    """

else:
    model = CAGFace(1).cuda()
print(" --- model loaded --- ")
print(model)
criteria = nn.SmoothL1Loss()
#criteria = nn.L1Loss()

## optimizer
#optimizer = optim.SGD(model.parameters(), lr=lr_start)
optimizer = optim.Adam(model.parameters(), lr=lr_start)



for epoch in range(10):
    iter = 0
    loss_lowest = 9999
    loss_avg = []
    for im, lb_512, lb_1024 in tqdm(dl):

        im = im.cuda()
        out = model(im)

        loss = criteria(lb_512,out.cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_avg.append(loss.item())

        iter += 1

        if(iter%10 == 0):
            print("iter  : ",iter)
            l = sum(loss_avg) / len(loss_avg)
            print("loss  : ",l)

            outImage = out.data.cpu()

            if visualize:
                plt.figure(1)
                plt.subplot(211)
                plt.imshow(transforms.ToPILImage()(outImage.squeeze()))
                plt.subplot(212)
                plt.imshow(transforms.ToPILImage()(lb_512.cpu().squeeze()))
                plt.pause(1)
                plt.close("all")

            if(l < loss_lowest):
                loss_lowest = l
                torch.save(model, "./checkpoints/"+str(epoch)+".pth")
                print("improved!")

            else:
                torch.save(model, "./checkpoints/"+str(epoch)+"_update"+".pth")

    print("epoch : ",epoch," \nloss : ",sum(loss_avg) / len(loss_avg))
    torch.save(model, "./checkpoints/"+str(epoch)+"_final"+".pth")
