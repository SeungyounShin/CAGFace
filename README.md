# CAGFace

Component Attention Guided Face Super-Resolution Network: CAGFace

https://arxiv.org/pdf/1910.08761.pdf

![x2_result](https://github.com/SeungyounShin/CAGFace/blob/master/results/Figure_1.png)

TODO
- [x] x2 network
- [ ] improve dataloader and preprocessin
- [ ] x4 network 
- [ ] evaluation

### Requirement
* pytorch 1.0 (checked at 1.0) 
* matplotlib
* Python3

### Prepare data 
Flickr-Faces-HQ dataset is used. 

### Train
You can use 
``` 
python train.py 
``` 

Refer the train.py files to check the arguement.

### References
* [CAGFace: Component Attention Guided Face Super-Resolution Network: CAGFace](https://arxiv.org/pdf/1910.08761.pdf)
* [Bisenet.pytorch](https://github.com/CoinCheung/BiSeNet)
