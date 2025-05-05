import collections
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import os

print(torch.version.cuda) 
t3 = time.time()

##############################################################################

args = {}
args['dim_h'] = 64         # factor controlling size of hidden layers
args['n_channel'] = 1 #3   # number of channels in the input data 
args['n_z'] = 300 #600     # number of dimensions in latent space. 
args['sigma'] = 1.0        # variance in n_z
args['lambda'] = 0.01      # hyper param for weight of discriminator loss
args['lr'] = 0.0002        # learning rate for Adam optimizer .000
args['epochs'] = 20        # how many epochs to run for
args['batch_size'] = 100   # batch size for SGD
args['save'] = True        # save weights at each epoch of training if True
args['train'] = True       # train networks if True, else load networks from
args['dataset'] = 'mnist'  #'fmnist' # specify which dataset to use

##############################################################################

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
        self.head = nn.Linear( args['n_z'], 10)   
    def forward(self, x):
        z = self.fc(self.conv(x).view(x.size(0), -1))
        return z, self.head(z)


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

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

def biased_get_class(c):
    xbeg = dec_x[dec_y == c]
    ybeg = dec_y[dec_y == c]
    return xbeg, ybeg

def G_SM(X, y, n_to_sample, cl, k=6):
    """
    Geodesic-Weighted SMOTE
    """
    k = min(k, len(X))          
    if k < 2:                  
        return X[:n_to_sample].copy(), [cl]*n_to_sample
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    nn = NearestNeighbors(n_neighbors=k, n_jobs=1).fit(Xn)
    _, ind = nn.kneighbors(Xn)
    base_idx  = np.random.choice(len(Xn), n_to_sample)
    neigh_off = np.random.randint(1, k, n_to_sample)
    neigh_idx = ind[base_idx, neigh_off]
    Vb, Vn = Xn[base_idx], Xn[neigh_idx]
    u = np.random.rand(n_to_sample, 1)
    cos_t = np.clip((Vb * Vn).sum(1, keepdims=True), -1., 1.)
    theta = np.arccos(cos_t)
    s = np.sin(theta) + 1e-12
    Vs = (np.sin((1-u)*theta)/s) * Vb + (np.sin(u*theta)/s) * Vn
    w = np.exp(-theta)         
    Vs = w * Vs + (1 - w) * Vb
    return Vs.astype(np.float32), [cl] * n_to_sample

dtrnimg = 'MNIST/trn_img/'
dtrnlab = 'MNIST/trn_lab/'

ids = os.listdir(dtrnimg)
idtri_f = [os.path.join(dtrnimg, image_id) for image_id in ids]
print(idtri_f)
ids = os.listdir(dtrnlab)
idtrl_f = [os.path.join(dtrnlab, image_id) for image_id in ids]
print(idtrl_f)

acc_history = []
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
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    trnimgfile = idtri_f[i]
    trnlabfile = idtrl_f[i]
    dec_x = np.loadtxt(trnimgfile) 
    dec_y = np.loadtxt(trnlabfile)
    print('train imgs before reshape ',dec_x.shape) 
    print('train labels ',dec_y.shape) 
    print(collections.Counter(dec_y))
    dec_x = dec_x.reshape(dec_x.shape[0],1,28,28)   
    print('train imgs after reshape ',dec_x.shape) 
    batch_size = 100
    num_workers = 0
    tensor_x = torch.Tensor(dec_x)
    tensor_y = torch.tensor(dec_y,dtype=torch.long)
    mnist_bal = TensorDataset(tensor_x,tensor_y) 
    train_loader = torch.utils.data.DataLoader(mnist_bal, 
        batch_size=batch_size,shuffle=True,num_workers=num_workers)
    
    classes = ('0', '1', '2', '3', '4','5', '6', '7', '8', '9')
    best_loss = np.inf
    acc_epoch_list = [] 
    t0 = time.time()
    if args['train']:
        opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args['lr'])
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.5)
    
        for epoch in range(args['epochs']):
            train_loss = 0.0
            tmse_loss = 0.0
            tdiscr_loss = 0.0
            encoder.train()
            decoder.train()
            for images,labs in train_loader:
                encoder.zero_grad()
                decoder.zero_grad()
                images, labs = images.to(device), labs.to(device)
                labsn = labs.detach().cpu().numpy()
                z_hat, logits = encoder(images)
                x_hat = decoder(z_hat) 
                mse = criterion(x_hat,images)
                z_min = None
                for _ in range(5):
                    tc = np.random.randint(0, 10)                       
                    mask = (labs == tc)
                    if mask.sum() > 1:
                        z_min = z_hat[mask].detach().cpu().numpy()        
                        break                                             
                if z_min is None:
                    mse2 = torch.tensor(0., device=device)
                else:
                    nsamp = min(len(z_min), 100)                         
                    z_syn_np, _ = G_SM(z_min, None, nsamp, cl=tc)         
                    z_syn = torch.tensor(z_syn_np, device=device)
                    x_syn = decoder(z_syn)        
                    base_imgs = images[mask][:nsamp]                      
                    mse2 = criterion(x_syn, base_imgs)                    
                loss_ce = nn.CrossEntropyLoss()(logits, labs)
                comb_loss = mse + 0.2 * mse2 + loss_ce
                comb_loss.backward()
                opt.step()
                sched.step()
                train_loss += comb_loss.item()*images.size(0)
                tmse_loss += mse.item()*images.size(0)
                tdiscr_loss += mse2.item()*images.size(0)
            train_loss = train_loss/len(train_loader)
            tmse_loss = tmse_loss/len(train_loader)
            tdiscr_loss = tdiscr_loss/len(train_loader)
            print('Epoch: {} \tTrain Loss: {:.6f} \tmse loss: {:.6f} \tmse2 loss: {:.6f}'.format(epoch,train_loss,tmse_loss,tdiscr_loss))
            
            encoder.eval()
            if epoch == 0:
                lin_cls = nn.Linear(args['n_z'], 10).to(device)
                opt_lin  = torch.optim.SGD(lin_cls.parameters(), lr=0.1)
                crit_ce  = nn.CrossEntropyLoss()

            encoder.eval()
            lin_cls = nn.Linear(args['n_z'], 10).to(device)
            opt_lin = torch.optim.SGD(lin_cls.parameters(), lr=0.1)
            crit_ce = nn.CrossEntropyLoss()

            for _ in range(10):                                 
                for img, lab in train_loader:
                    with torch.no_grad():
                        z, _ = encoder(img.to(device))
                    opt_lin.zero_grad()
                    loss = crit_ce(lin_cls(z), lab.to(device))
                    loss.backward(); opt_lin.step()

            correct = total = 0
            with torch.no_grad():
                for img, lab in train_loader:
                    z, _ = encoder(img.to(device))
                    pred = lin_cls(z).argmax(1).cpu()
                    correct += (pred == lab).sum().item()
                    total += lab.size(0)
            final_acc = correct / total
            print(f"Final train accuracy: {final_acc:.4f}")
            acc_epoch_list.append(final_acc)          

            if train_loss < best_loss:
                print('Saving..')
                path_enc = 'MNIST/models/crs5/' \
                    + str(i) + '/bst_enc.pth'
                path_dec = 'MNIST/models/crs5/' \
                    + str(i) + '/bst_dec.pth'
             
                torch.save(encoder.state_dict(), path_enc)
                torch.save(decoder.state_dict(), path_dec)
                best_loss = train_loss

        if acc_epoch_list:                                   
            mean_acc_epochs = sum(acc_epoch_list) / len(acc_epoch_list)
            print(f"Mean accuracy over {len(acc_epoch_list)} epochs: {mean_acc_epochs:.4f}")

        path_enc = 'MNIST/models/crs5/' \
            + str(i) + '/f_enc.pth'
        path_dec = 'MNIST/models/crs5/' \
            + str(i) + '/f_dec.pth'
        print(path_enc)
        print(path_dec)
        torch.save(encoder.state_dict(), path_enc)
        torch.save(decoder.state_dict(), path_dec)
        print()
    t1 = time.time()
    print('total time(min): {:.2f}'.format((t1 - t0)/60))             
 
t4 = time.time()
print('final time(min): {:.2f}'.format((t4 - t3)/60))