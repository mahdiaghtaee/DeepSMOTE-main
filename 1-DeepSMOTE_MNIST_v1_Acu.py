import collections
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import os

print(torch.version.cuda) #10.1
t3 = time.time()
##############################################################################
"""args for AE"""

args = {}
args['dim_h'] = 64         # factor controlling size of hidden layers
args['n_channel'] = 1      # number of channels in the input data 
args['n_z'] = 300          # number of dimensions in latent space.
args['sigma'] = 1.0        # variance in n_z
args['lambda'] = 0.01      # hyper param for weight of discriminator loss
args['lr'] = 0.0002        # learning rate for Adam optimizer .000
args['epochs'] = 2       # how many epochs to run for
args['batch_size'] = 100   # batch size for SGD
args['save'] = True        # save weights at each epoch of training if True
args['train'] = True       # train networks if True, else load networks from
args['dataset'] = 'mnist'  # specify which dataset to use

##############################################################################

## create encoder model and decoder model
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        
        # convolutional filters, work excellent with image data
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
            nn.LeakyReLU(0.2, inplace=True)
        )
        # final layer is fully connected
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

        # first layer is fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * 7 * 7),
            nn.ReLU())

        # deconvolutional filters, essentially inverse of convolutional filters
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

##############################################################################
"""MLP classifier for computing Accuracy"""

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10, hidden_dim=128):
        super(MLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

##############################################################################
"""functions to create SMOTE images"""

def biased_get_class(c):
    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]
    return xbeg, ybeg

def G_SM(X, y, n_to_sample, cl):
    # determining the number of samples to generate
    n_neigh = 5 + 1
    nn_obj = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    nn_obj.fit(X)
    dist, ind = nn_obj.kneighbors(X)

    # generating samples
    base_indices = np.random.choice(list(range(len(X))), n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)
    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]
    samples = X_base + np.multiply(np.random.rand(n_to_sample, 1),
            X_neighbor - X_base)
    # use 10 as label because 0 to 9 real classes and 1 fake/smoted = 10
    return samples, [cl]*n_to_sample

##############################################################################
#NOTE: Download the training ('.../0_trn_img.txt') and label files 
# ('.../0_trn_lab.txt').  Place the files in directories (e.g., ../MNIST/trn_img/
# and /MNIST/trn_lab/).

dtrnimg = 'MNIST/trn_img/'
dtrnlab = 'MNIST/trn_lab/'

ids = os.listdir(dtrnimg)
idtri_f = [os.path.join(dtrnimg, image_id) for image_id in ids]
print(idtri_f)

ids = os.listdir(dtrnlab)
idtrl_f = [os.path.join(dtrnlab, image_id) for image_id in ids]
print(idtrl_f)

#for i in range(5):
for i in range(len(ids)):
    print()
    print(i)
    encoder = Encoder(args)
    decoder = Decoder(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    train_on_gpu = torch.cuda.is_available()

    #decoder loss function
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    
    trnimgfile = idtri_f[i]
    trnlabfile = idtrl_f[i]
    
    print(trnimgfile)
    print(trnlabfile)
    dec_x = np.loadtxt(trnimgfile) 
    dec_y = np.loadtxt(trnlabfile)

    print('train imgs before reshape ', dec_x.shape) 
    print('train labels ', dec_y.shape) 
    print(collections.Counter(dec_y))
    dec_x = dec_x.reshape(dec_x.shape[0], 1, 28, 28)   
    print('train imgs after reshape ', dec_x.shape) 

    batch_size = args['batch_size']
    num_workers = 0

    tensor_x = torch.Tensor(dec_x)
    tensor_y = torch.tensor(dec_y, dtype=torch.long)
    mnist_bal = TensorDataset(tensor_x, tensor_y) 
    train_loader = torch.utils.data.DataLoader(mnist_bal, 
        batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    classes = ('0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9')

    best_loss = np.inf

    t0 = time.time()
    if args['train']:
        enc_optim = torch.optim.Adam(encoder.parameters(), lr=args['lr'])
        dec_optim = torch.optim.Adam(decoder.parameters(), lr=args['lr'])
    
        for epoch in range(args['epochs']):
            train_loss = 0.0
            tmse_loss = 0.0
            tdiscr_loss = 0.0
            encoder.train()
            decoder.train()
        
            for images, labs in train_loader:
                encoder.zero_grad()
                decoder.zero_grad()
                images, labs = images.to(device), labs.to(device)
                z_hat = encoder(images)
                x_hat = decoder(z_hat)
                mse = criterion(x_hat, images)
                
                # محاسبه loss اضافی بر روی یک کلاس نمونه
                tc = np.random.choice(10, 1)
                xbeg = dec_x[dec_y == tc]
                ybeg = dec_y[dec_y == tc] 
                xlen = len(xbeg)
                nsamp = min(xlen, 100)
                ind = np.random.choice(list(range(len(xbeg))), nsamp, replace=False)
                xclass = xbeg[ind]
                yclass = ybeg[ind]
                xclen = len(xclass)
                xcminus = np.arange(1, xclen)
                xcplus = np.append(xcminus, 0)
                xcnew = (xclass[[xcplus], :])
                xcnew = xcnew.reshape(xcnew.shape[1], xcnew.shape[2], xcnew.shape[3], xcnew.shape[4])
                xcnew = torch.Tensor(xcnew).to(device)
                xclass = torch.Tensor(xclass).to(device)
                xclass = encoder(xclass)
                xclass = xclass.detach().cpu().numpy()
                xc_enc = (xclass[[xcplus], :])
                xc_enc = np.squeeze(xc_enc)
                xc_enc = torch.Tensor(xc_enc).to(device)
                ximg = decoder(xc_enc)
                mse2 = criterion(ximg, xcnew)
            
                comb_loss = mse2 + mse
                comb_loss.backward()
            
                enc_optim.step()
                dec_optim.step()
            
                train_loss += comb_loss.item() * images.size(0)
                tmse_loss += mse.item() * images.size(0)
                tdiscr_loss += mse2.item() * images.size(0)
            
            train_loss = train_loss / len(train_loader)
            tmse_loss = tmse_loss / len(train_loader)
            tdiscr_loss = tdiscr_loss / len(train_loader)
            print('Epoch: {} \tTrain Loss: {:.6f} \tmse loss: {:.6f} \tmse2 loss: {:.6f}'.format(epoch,
                    train_loss, tmse_loss, tdiscr_loss))
            
            # ذخیره بهترین مدل
            if train_loss < best_loss:
                print('Saving..')
                path_enc = 'MNIST/models/crs5/' + str(i) + '/bst_enc.pth'
                path_dec = 'MNIST/models/crs5/' + str(i) + '/bst_dec.pth'
                torch.save(encoder.state_dict(), path_enc)
                torch.save(decoder.state_dict(), path_dec)
                best_loss = train_loss
        
        # ذخیره مدل نهایی
        path_enc = 'MNIST/models/crs5/' + str(i) + '/f_enc.pth'
        path_dec = 'MNIST/models/crs5/' + str(i) + '/f_dec.pth'
        print(path_enc)
        print(path_dec)
        torch.save(encoder.state_dict(), path_enc)
        torch.save(decoder.state_dict(), path_dec)
        print()
              
    t1 = time.time()
    print('total time(min): {:.2f}'.format((t1 - t0)/60))    

    acc_list = []  # لیست برای ذخیره دقت هر epoch         

    ##############################################################################
    """آموزش MLP برای محاسبه دقت (Accuracy) با استفاده از خروجی رمزگذار"""
    # تنظیم encoder در حالت ارزیابی
    encoder.eval()
    # ایجاد نمونه MLPClassifier
    mlp = MLPClassifier(input_dim=args['n_z'], num_classes=10).to(device)
    criterion_cls = nn.CrossEntropyLoss().to(device)
    optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=args['lr'])
    num_epochs_mlp = 20  # تعداد epoch برای آموزش MLP

    for epoch in range(num_epochs_mlp):
        running_loss = 0.0
        correct = 0
        total = 0
        mlp.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                latent = encoder(images)
            outputs = mlp(latent)
            loss_cls = criterion_cls(outputs, labels)
            optimizer_mlp.zero_grad()
            loss_cls.backward()
            optimizer_mlp.step()
            running_loss += loss_cls.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        acc_list.append(epoch_acc)  # ذخیره دقت در لیست
        print('MLP Epoch: {} \tLoss: {:.4f} \tAccuracy: {:.4f}'.format(epoch, epoch_loss, epoch_acc))
    ##############################################################################
    
    
    mean_accuracy = sum(acc_list) / len(acc_list)
    print('Mean Accuracy over {} epochs: {:.4f}'.format(num_epochs_mlp, mean_accuracy))
    t4 = time.time()
    print('final time(min): {:.2f}'.format((t4 - t3)/60))
