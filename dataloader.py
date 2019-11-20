from torchvision import transforms
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F
import os
from sklearn.preprocessing import normalize
from PIL import Image

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

class Flicker():
    def __init__(self, img_path ,transform_img ,transform_stage1,transform , resize, mode=True):
        self.img_path = img_path
        self.transform_img = transform_img
        self.transform_stage1 = transform_stage1
        self.transform_label = transform
        self.normalize = transforms.Compose([transforms.ToPILImage(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.smoothing = GaussianSmoothing(3,5,1).cpu()
        self.train_dataset = []
        self.mode = mode
        self.resize = resize
        self.bisenet = torch.load('./weights/bisenet.pth').cpu().eval()

        self.preprocess()

        self.num_images = len(self.train_dataset)

    def preprocess(self):

        dir = os.listdir(self.img_path)
        if(self.mode==True):
            for i in dir:
                for j in os.listdir(self.img_path+'/'+i):
                    self.train_dataset.append(self.img_path+'/'+i+'/'+j)
        else:
            for i in dir:
                self.train_dataset.append(self.img_path+'/'+i)

        print('Finished preprocessing the Flicker dataset...')

    def __getitem__(self, index):

        img_path = self.train_dataset[index]
        image = Image.open(img_path)
        H,W = image.size
        imageTensor = self.transform_img(image).view(1,3,self.resize, self.resize).float()
        inputShape  = imageTensor.size()
        out , out16, out32 = self.bisenet(imageTensor)
        out   = out.max(dim=1)[0]
        out16 = out16.max(dim=1)[0]
        out32 = out32.max(dim=1)[0]
        prior = torch.cat((out,out16,out32), dim=0).view(inputShape)

        prior = F.pad(prior, (2, 2, 2, 2), mode='reflect')
        with torch.no_grad():
            prior = self.smoothing(prior)


        prior = self.normalize(prior.squeeze()).unsqueeze(0)
        input = torch.cat((imageTensor, prior), dim=1)

        return input.squeeze(), self.transform_stage1(image) , self.transform_label(image)

    def __len__(self):
        """Return the number of images."""
        return self.num_images

class Data_Loader():
    def __init__(self, img_path, resize, batch_size, mode=True):
        self.img_path = img_path
        self.resize = resize
        self.batch = batch_size
        self.mode = mode

    def transform_img(self, resize, totensor, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160))
        if resize:
            options.append(transforms.Resize((self.resize,self.resize)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def transform_img2(self, resize, totensor):
        options = []
        if resize:
            options.append(transforms.Resize((self.resize*2,self.resize*2)))
        if totensor:
            options.append(transforms.ToTensor())
        transform = transforms.Compose(options)
        return transform

    def transform_label(self, totensor):
        options = []
        if totensor:
            options.append(transforms.ToTensor())
        transform = transforms.Compose(options)
        return transform

    def loader(self):
        img = self.transform_img(True, True, True, False)
        label0 = self.transform_img2(True,True)
        label = self.transform_label(True)
        dataset = Flicker(self.img_path, img,label0,  label, self.resize,self.mode)

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             shuffle=True,
                                             num_workers=2,
                                             drop_last=False)
        return loader

if __name__ == '__main__':
    dl = Data_Loader('/home/yo0n/바탕화면/Flicker',512,2).loader()
    for im, lb in dl:
        print("dl loop : ",im.size(), lb.size())
