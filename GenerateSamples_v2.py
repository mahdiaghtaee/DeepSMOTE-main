# اصلاح ساختار آدرس دهی
# strict=True
# np.save / np.load 


# -*- coding: utf-8 -*-

import collections
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
print(torch.version.cuda) #10.1
import time
t0 = time.time()
##############################################################################
"""args for models"""

args = {}
#تعداد نرون‌ها در لایه‌های پنهان شبکه.
args['dim_h'] = 64          # factor controlling size of hidden layers
 # تعداد کانال‌های ورودی (در اینجا 1 برای تصاویر سیاه‌وسفید)
args['n_channel'] = 1       # number of channels in the input data 
# تعداد ابعاد فضای نهان (latent space)
args['n_z'] = 300 #600     # number of dimensions in latent space. 
# واریانس در فضای نهان
args['sigma'] = 1.0        # variance in n_z
 # پارامتر تنظیمی برای وزن دادن به ضرر Discriminator
args['lambda'] = 0.01      # hyper param for weight of discriminator loss
# نرخ یادگیری برای الگوریتم Adam
args['lr'] = 0.0002        # learning rate for Adam optimizer .000
 # تعداد اپوک‌ها برای آموزش
args['epochs'] = 1 #50         # how many epochs to run for
# اندازه بچ (Batch Size)
args['batch_size'] = 100   # batch size for SGD
 # اگر True باشد، وزن‌ها در هر اپوک ذخیره می‌شود
args['save'] = True        # save weights at each epoch of training if True
  # اگر True باشد، مدل آموزش داده می‌شود، در غیر این صورت مدل بارگذاری می‌شود
args['train'] = True       # train networks if True, else load networks from
# دیتاست مورد استفاده (در اینجا MNIST)
args['dataset'] = 'mnist' #'fmnist' # specify which dataset to use

##############################################################################

## create encoder model and decoder model
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        
        # لایه‌های کانولوشن برای استخراج ویژگی‌ها از تصاویر
        # convolutional filters, work excellent with image data
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True),
            
            
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            
            #3d and 32 by 32
            #nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 1, 0, bias=False),
            
            nn.BatchNorm2d(self.dim_h * 8), # 40 X 8 = 320
            #nn.ReLU(True),
            nn.LeakyReLU(0.2, inplace=True) )#,
            #nn.Conv2d(self.dim_h * 8, 1, 2, 1, 0, bias=False))
            #nn.Conv2d(self.dim_h * 8, 1, 4, 1, 0, bias=False))
        # final layer is fully connected
        # لایه fully connected در انتها برای فشرده‌سازی ویژگی‌ها به فضای نهان
        self.fc = nn.Linear(self.dim_h * (2 ** 3), self.n_z)
        

    # عبور داده‌ها از لایه‌های کانولوشن
    # حذف ابعاد اضافی
     # عبور از لایه fully connected برای تولید ویژگی‌های فضای نهان
    def forward(self, x):
        #print('enc')
        #print('input ',x.size()) #torch.Size([100, 3,32,32])
        x = self.conv(x)
        #print('aft conv ',x.size()) #torch.Size([100, 320, 2, 2]) with 
        #nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
        #vs torch.Size([128, 320, 1, 1])
        #aft conv  torch.Size([100, 320, 1, 1]) with 
        #nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 1, 0, bias=False),
        x = x.squeeze()
        #print('aft squeeze ',x.size()) #torch.Size([128, 320])
        #aft squeeze  torch.Size([100, 320])
        x = self.fc(x)
        #print('out ',x.size()) #torch.Size([128, 20])
        #out  torch.Size([100, 300])
        return x

# مدل Decoder برای بازسازی داده‌ها از فضای نهان
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        # اولین لایه Fully Connected برای تبدیل ورودی از فضای نهان به فضای بزرگ‌تر
        # first layer is fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.ReLU())

        # لایه‌های Deconvolution برای بازسازی تصاویر از ویژگی‌های فضای نهان
        # deconvolutional filters, essentially inverse of convolutional filters
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, 1, 4, stride=2),
            #nn.Sigmoid()
            nn.Tanh())

    def forward(self, x):
        #print('dec')
        #print('input ',x.size())
        x = self.fc(x)
        x = x.view(-1, self.dim_h * 8, 7, 7)
        x = self.deconv(x)
        return x

##############################################################################

# تابع برای دریافت داده‌های کلاس خاص (مثلاً یک کلاس خاص از داده‌ها)
def biased_get_class1(c):
    
    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]
    
    return xbeg, ybeg
    #return xclass, yclass

# تابع برای تولید نمونه‌های مصنوعی از داده‌های کلاس خاص
def G_SM1(X, y,n_to_sample,cl):
    # فیت کردن مدل Nearest Neighbors برای انتخاب نزدیک‌ترین نمونه‌ها
    # fitting the model
    n_neigh = 5 + 1
    nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)

    # generating samples
    base_indices = np.random.choice(list(range(len(X))),n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)),n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    samples = X_base + np.multiply(np.random.rand(n_to_sample,1),
            X_neighbor - X_base)

    # استفاده از 10 به عنوان برچسب برای نمونه‌های مصنوعی
    #use 10 as label because 0 to 9 real classes and 1 fake/smoted = 10
    return samples, [cl]*n_to_sample

#############################################################################
 # تنظیمات چاپ اعداد برای دقت بیشتر
np.printoptions(precision=5,suppress=True)

# لیست مسیرهای فایل‌ها برای داده‌های آموزش
# Define the base directory for images and labels
base_trnimg_dir = 'MNIST/trn_img/'  # Adjust as per your actual directory structure
base_trnlab_dir = 'MNIST/trn_lab/'  # Adjust as per your actual directory structure

# تعداد مدل‌ها
# Assuming you have 5 sets (0 to 4)
num_models = 5

# Lists to store file paths
idtri_f = [os.path.join(base_trnimg_dir, f"{m}_trn_img.txt") for m in range(num_models)]
idtrl_f = [os.path.join(base_trnlab_dir, f"{m}_trn_lab.txt") for m in range(num_models)]

print("Image Files:", idtri_f)
print("Label Files:", idtrl_f)

# مسیر ذخیره‌سازی مدل‌ها
#path on the computer where the models are stored
modpth = 'MNIST/models/crs5/'

encf = []
decf = []
for p in range(5):
    enc = os.path.join(modpth, str(p), 'bst_enc.pth')
    dec = os.path.join(modpth, str(p), 'bst_dec.pth')
    encf.append(enc)
    decf.append(dec)

# پردازش داده‌ها برای هر مدل
for m in range(5):
    print(m)
    trnimgfile = idtri_f[m]
    trnlabfile = idtrl_f[m]
    print(trnimgfile)
    print(trnlabfile)

    # بارگذاری داده‌های آموزشی
    dec_x = np.loadtxt(trnimgfile) 
    dec_y = np.loadtxt(trnlabfile)

    # نمایش ابعاد تصاویر قبل از تغییر شکل
    print('train imgs before reshape ',dec_x.shape) #(44993, 3072) 45500, 3072)
    print('train labels ',dec_y.shape) #(44993,) (45500,)

    # تغییر شکل داده‌ها به ابعاد مناسب برای پردازش
    dec_x = dec_x.reshape(dec_x.shape[0],1,28,28)

    print('decy ',dec_y.shape)
    print(collections.Counter(dec_y))
    
    print('train imgs after reshape ',dec_x.shape) #(45000,3,32,32)

    classes = ('0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9')
    
    #generate some images 
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

    #imbal = [4500, 2000, 1000, 800, 600, 500, 400, 250, 150, 80]
    imbal = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]

    resx = []
    resy = []

    # برای هر کلاس داده‌های مصنوعی تولید می‌شود
    for i in range(1,10):
         # دریافت داده‌ها و برچسب‌های کلاس i
        xclass, yclass = biased_get_class1(i)
        # چاپ ابعاد داده‌ها
        print(xclass.shape) #(500, 3, 32, 32)
        print(yclass[0]) #(500,)
            
        # کدگذاری داده‌های xclass به فضای ویژگی‌ها    
        #encode xclass to feature space
        xclass = torch.Tensor(xclass)
        xclass = xclass.to(device)
        xclass = encoder(xclass)
        print(xclass.shape) #torch.Size([500, 600])
            
        # تبدیل داده‌ها به numpy array    
        xclass = xclass.detach().cpu().numpy()
        n = imbal[0] - imbal[i]
        xsamp, ysamp = G_SM1(xclass,yclass,n,i)
        print(xsamp.shape) #(4500, 600)
        print(len(ysamp)) #4500
        ysamp = np.array(ysamp)
        print(ysamp.shape) #4500   
    
        # تولید تصاویر از نمونه‌های مصنوعی برای استفاده در مدل ResNet
        """to generate samples for resnet"""   
        xsamp = torch.Tensor(xsamp)
        xsamp = xsamp.to(device)
        #xsamp = xsamp.view(xsamp.size()[0], xsamp.size()[1], 1, 1)
        #print(xsamp.size()) #torch.Size([10, 600, 1, 1])
        ximg = decoder(xsamp)

        ximn = ximg.detach().cpu().numpy()
        print(ximn.shape) #(4500, 3, 32, 32)
        #ximn = np.expand_dims(ximn,axis=1)
        print(ximn.shape) #(4500, 3, 32, 32)
        resx.append(ximn)
        resy.append(ysamp)
        #print('resx ',resx.shape)
        #print('resy ',resy.shape)
        #print()
    
    # ترکیب تمام تصاویر و برچسب‌ها
    resx1 = np.vstack(resx)
    resy1 = np.hstack(resy)
    #print(resx1.shape) #(34720, 3, 32, 32)
    #resx1 = np.squeeze(resx1)
    print(resx1.shape) #(34720, 3, 32, 32)
    print(resy1.shape) #(34720,)

    resx1 = resx1.reshape(resx1.shape[0],-1)
    print(resx1.shape) #(34720, 3072)
    
    dec_x1 = dec_x.reshape(dec_x.shape[0],-1)
    print('decx1 ',dec_x1.shape)
    combx = np.vstack((resx1,dec_x1))
    comby = np.hstack((resy1,dec_y))

    print(combx.shape) #(45000, 3, 32, 32)
    print(comby.shape) #(45000,)

    ifile = 'MNIST/trn_img_f/' + \
        str(m) + '_trn_img.txt'
    np.savetxt(ifile, combx)
    
    lfile = 'MNIST/trn_lab_f/' + \
        str(m) + '_trn_lab.txt'
    np.savetxt(lfile,comby) 
    print()

t1 = time.time()
print('final time(min): {:.2f}'.format((t1 - t0)/60))
