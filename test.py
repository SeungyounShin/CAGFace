import torch
from dataloader import Data_Loader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

#model = torch.load('./checkpoints/cand1.pth').cpu().eval()
model = torch.load('/home/yo0n/바탕화면/CAGFace/checkpoints/noBatchNorm/cand3.pth').cpu().eval()

dl = Data_Loader('/home/yo0n/바탕화면/CAGFace/test/',256,1,False).loader()

result = list()

def show(img):
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    print(npimg.shape)
    plt.imshow(npimg)
    plt.show()

for im,lb in dl:
    #im = im.cuda()
    out = model(im)
    result.append(out.detach().squeeze())

    plt.imshow(transforms.ToPILImage()(out.squeeze()))
    plt.show()
    plt.imshow(transforms.ToPILImage()(lb.squeeze()))
    plt.show()

    print(out.size())

    plt.figure(1)
    plt.subplot(211)
    histOut = [i*0 for i in range(285)]
    for i in list((out.data*255).long().view(-1).numpy()):
        histOut[i] += 1
    plt.bar(np.arange(len(histOut)),histOut)
    plt.subplot(212)
    histlb = [i*0 for i in range(255)]
    for i in list((lb.data*255).long().view(-1).numpy()):
        histlb[i] += 1
    plt.bar(np.arange(len(histlb)),histlb)
    plt.show()


#show(result[0])
