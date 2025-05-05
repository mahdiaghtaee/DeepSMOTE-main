import collections
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
print(torch.version.cuda) 
import time
t0 = time.time()

args = {}
args['dim_h'] = 64          # factor controlling size of hidden layers
args['n_channel'] = 1       # number of channels in the input data 
args['n_z'] = 300 #600     # number of dimensions in latent space. 
args['sigma'] = 1.0        # variance in n_z
args['lambda'] = 0.01      # hyper param for weight of discriminator loss
args['lr'] = 0.0002        # learning rate for Adam optimizer .000
args['epochs'] = 1 #50         # how many epochs to run for
args['batch_size'] = 100   # batch size for SGD
args['save'] = True        # save weights at each epoch of training if True
args['train'] = True       # train networks if True, else load networks from
args['dataset'] = 'mnist' #'fmnist' # specify which dataset to use

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8), 
            nn.LeakyReLU(0.2, inplace=True) )#,
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze()
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.ReLU())

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, 1, 4, stride=2),
            nn.Tanh())

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.deconv(x)
        return x

def biased_get_class1(c):
    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]
    return xbeg, ybeg

def G_SM1(X, y,n_to_sample,cl):
    n_neigh = 5 + 1
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)
    base_indices = np.random.choice(list(range(len(X))),n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)),n_to_sample)
    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]
    samples = X_base + np.multiply(np.random.rand(n_to_sample,1),
            X_neighbor - X_base)

    return samples, [cl]*n_to_sample

np.printoptions(precision=5,suppress=True)
base_trnimg_dir = 'MNIST/trn_img/'   
base_trnlab_dir = 'MNIST/trn_lab/'   

num_models = 5
idtri_f = [os.path.join(base_trnimg_dir, f"{m}_trn_img.txt") for m in range(num_models)]
idtrl_f = [os.path.join(base_trnlab_dir, f"{m}_trn_lab.txt") for m in range(num_models)]

print("Image Files:", idtri_f)
print("Label Files:", idtrl_f)

modpth = 'MNIST/models/crs5/'

encf = []
decf = []
for p in range(5):
    enc = os.path.join(modpth, str(p), 'bst_enc.pth')
    dec = os.path.join(modpth, str(p), 'bst_dec.pth')
    encf.append(enc)
    decf.append(dec)

for m in range(5):
    print(m)
    trnimgfile = idtri_f[m]
    trnlabfile = idtrl_f[m]
    print(trnimgfile)
    print(trnlabfile)

    dec_x = np.loadtxt(trnimgfile) 
    dec_y = np.loadtxt(trnlabfile)

    print('train imgs before reshape ',dec_x.shape)  
    print('train labels ',dec_y.shape)  

    dec_x = dec_x.reshape(dec_x.shape[0],1,28,28)

    print('decy ',dec_y.shape)
    print(collections.Counter(dec_y))
    
    print('train imgs after reshape ',dec_x.shape)  

    classes = ('0', '1', '2', '3', '4','5', '6', '7', '8', '9')
    
    train_on_gpu = torch.cuda.is_available()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    path_enc = encf[m]
    path_dec = decf[m]
    encoder = Encoder(args)
    encoder.load_state_dict(torch.load(path_enc), strict=True)
    encoder = encoder.to(device)
    decoder = Decoder(args)
    decoder.load_state_dict(torch.load(path_dec), strict=True)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    imbal = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]

    resx = []
    resy = []

    for i in range(1,10):
        xclass, yclass = biased_get_class1(i)
        print(xclass.shape)  
        print(yclass[0])  
        xclass = torch.Tensor(xclass)
        xclass = xclass.to(device)
        xclass = encoder(xclass)
        print(xclass.shape)  
        xclass = xclass.detach().cpu().numpy()
        n = imbal[0] - imbal[i]
        xsamp, ysamp = G_SM1(xclass,yclass,n,i)
        print(xsamp.shape)  
        print(len(ysamp))  
        ysamp = np.array(ysamp)
        print(ysamp.shape) 
        xsamp = torch.Tensor(xsamp)
        xsamp = xsamp.to(device)
        ximg = decoder(xsamp)
        ximn = ximg.detach().cpu().numpy()
        print(ximn.shape)  
        print(ximn.shape)  
        resx.append(ximn)
        resy.append(ysamp)
    
    resx1 = np.vstack(resx)
    resy1 = np.hstack(resy)
    print(resx1.shape)  
    print(resy1.shape)  
    resx1 = resx1.reshape(resx1.shape[0],-1)
    print(resx1.shape)  
    dec_x1 = dec_x.reshape(dec_x.shape[0],-1)
    print('decx1 ',dec_x1.shape)
    combx = np.vstack((resx1,dec_x1))
    comby = np.hstack((resy1,dec_y))
    print(combx.shape)  
    print(comby.shape)  
    ifile = 'MNIST/trn_img_f/' + \
        str(m) + '_trn_img.txt'
    np.savetxt(ifile, combx)
    
    lfile = 'MNIST/trn_lab_f/' + \
        str(m) + '_trn_lab.txt'
    np.savetxt(lfile,comby) 
    print()
t1 = time.time()
print('final time(min): {:.2f}'.format((t1 - t0)/60))
