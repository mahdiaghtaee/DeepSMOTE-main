import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from sklearn.neighbors import NearestNeighbors
import collections

##############################################################################
# هایپرپارامترها
##############################################################################
class Config:
    dim_h     = 64   # تعداد فیلترهای پایه در کانولوشن
    n_channel = 1    # تعداد کانال ورودی (Fashion MNIST = 1)
    n_z       = 300  # تعداد ابعاد فضای نهان
    lr        = 0.0002
    epochs    = 5    # تعداد تکرار آموزش (دلخواه)
    batch_size= 128

args = Config()

##############################################################################
# معماری Encoder و Decoder اصلاح‌شده
##############################################################################
# توجه: در لایهٔ آخر Encoder و لایهٔ اول Decoder از kernel=3,stride=3,pad=0
# یا از پارامترهای output_padding برای رسیدن به ابعاد دلخواه استفاده می‌کنیم.

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.dim_h = args.dim_h
        self.n_channel = args.n_channel
        self.n_z   = args.n_z

        # مراحل:
        # 1) 28->14  (kernel=4,stride=2,pad=1)
        # 2) 14->7
        # 3) 7->3
        # 4) 3->1   (kernel=3,stride=3,pad=0)
        self.conv = nn.Sequential(
            # (1,28,28) -> (64,14,14)
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h),
            nn.LeakyReLU(0.2, inplace=True),

            # (64,14,14) -> (128,7,7)
            nn.Conv2d(self.dim_h, self.dim_h*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h*2),
            nn.LeakyReLU(0.2, inplace=True),

            # (128,7,7) -> (256,3,3)
            nn.Conv2d(self.dim_h*2, self.dim_h*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h*4),
            nn.LeakyReLU(0.2, inplace=True),

            # (256,3,3) -> (512,1,1)
            nn.Conv2d(self.dim_h*4, self.dim_h*8, 3, 3, 0, bias=False),
            nn.BatchNorm2d(self.dim_h*8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # خروجی کانولوشن (512,1,1) => 512 ویژگی
        self.fc = nn.Linear(self.dim_h*8, self.n_z)

    def forward(self, x):
        x = self.conv(x)               # (N,512,1,1)
        x = x.view(x.size(0), -1)      # (N,512)
        x = self.fc(x)                 # (N,300)
        return x


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.dim_h = args.dim_h
        self.n_z   = args.n_z

        # مرحلهٔ خطی 300->512
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h*8),
            nn.ReLU(True)
        )

        # حالا باید عکس مراحل قبل را انجام دهیم:
        # 1->3 (kernel=3,stride=3,pad=0)
        # 3->7 (kernel=4,stride=2,pad=1, output_padding=1)
        # 7->14 (kernel=4,stride=2,pad=1)
        # 14->28 (kernel=4,stride=2,pad=1)
        self.deconv = nn.Sequential(
            # (512,1,1) -> (256,3,3)
            nn.ConvTranspose2d(self.dim_h*8, self.dim_h*4, 3, 3, 0),
            nn.BatchNorm2d(self.dim_h*4),
            nn.ReLU(True),

            # (256,3,3) -> (128,7,7)
            nn.ConvTranspose2d(self.dim_h*4, self.dim_h*2, 4, 2, 1,
                               output_padding=1),
            nn.BatchNorm2d(self.dim_h*2),
            nn.ReLU(True),

            # (128,7,7) -> (64,14,14)
            nn.ConvTranspose2d(self.dim_h*2, self.dim_h, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),

            # (64,14,14) -> (1,28,28)
            nn.ConvTranspose2d(self.dim_h, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)                         # (N,512)
        x = x.view(x.size(0), self.dim_h*8, 1, 1)   # (N,512,1,1)
        x = self.deconv(x)                     # (N,1,28,28)
        return x

##############################################################################
# تابع آموزش Autoencoder روی FashionMNIST (از صفر)
##############################################################################
def train_autoencoder(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(
        root='./data_fashion',
        train=True,
        download=True,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = Encoder(args).to(device)
    decoder = Decoder(args).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                           lr=args.lr)

    best_loss = float('inf')
    save_folder = './models_fixedAE'
    os.makedirs(save_folder, exist_ok=True)

    print("\n=== Training Autoencoder from scratch ===")
    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        total_loss = 0

        for images, _ in train_loader:
            images = images.to(device)  # (N,1,28,28)

            # Forward
            z = encoder(images)   # (N,300)
            recon = decoder(z)    # (N,1,28,28)

            loss = criterion(recon, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        epoch_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{args.epochs}] | Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(encoder.state_dict(), os.path.join(save_folder, "enc_fixed.pth"))
            torch.save(decoder.state_dict(), os.path.join(save_folder, "dec_fixed.pth"))
            print(f"   -> Best model saved (loss={best_loss:.6f})")

    print("Training finished.")
    print(f"Best reconstruction loss: {best_loss:.6f}")
    return os.path.join(save_folder, "enc_fixed.pth"), os.path.join(save_folder, "dec_fixed.pth")


##############################################################################
# تولید نمونه‌های مصنوعی با SMOTE در فضای نهان
##############################################################################

def G_SM1(X, y, n_to_sample, cl):
    """ نسخه سادهٔ SMOTE با KNN """
    n_neigh = 6
    neigh = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
    neigh.fit(X)
    dist, ind = neigh.kneighbors(X)

    base_indices = np.random.choice(list(range(len(X))), n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, n_neigh)), n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]

    # ایجاد نمونه تصادفی بین دو نقطه
    lam = np.random.rand(n_to_sample, 1)
    samples = X_base + lam * (X_neighbor - X_base)
    labels_fake = [cl]*n_to_sample
    return samples, labels_fake

##############################################################################
# تلفیق کامل کد (Train + SMOTE) به سبک فولد
##############################################################################
def main():
    t0 = time.time()

    # مرحله 1) آموزش مدل از ابتدا
    path_enc, path_dec = train_autoencoder(args)

    # مرحله 2) استفاده از مدل برای تولید نمونه‌های مصنوعی (درصورت نیاز)
    # این قسمت صرفاً الگوست، شبیه کد قبلی شما که از فولدها استفاده می‌کرد.

    # بارگذاری مدل آموزش‌دیده
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = Encoder(args).to(device)
    decoder = Decoder(args).to(device)

    encoder.load_state_dict(torch.load(path_enc, map_location=device))
    decoder.load_state_dict(torch.load(path_dec, map_location=device))
    encoder.eval()
    decoder.eval()

    # اگر فولدها و فایل‌هایتان را دارید، به این شکل ادامه دهید:
    base_trnimg_dir = 'FMNIST/trn_img/'
    base_trnlab_dir = 'FMNIST/trn_lab/'
    num_models = 5

    idtri_f = [os.path.join(base_trnimg_dir, f"{m}_trn_img.txt") for m in range(num_models)]
    idtrl_f = [os.path.join(base_trnlab_dir, f"{m}_trn_lab.txt") for m in range(num_models)]
    print("Image Files:", idtri_f)
    print("Label Files:", idtrl_f)

    imbal = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]

    for m in range(num_models):
        print(f"\nProcessing fold: {m}")
        trnimgfile = idtri_f[m]
        trnlabfile = idtrl_f[m]
        print("Train images file:", trnimgfile)
        print("Train labels file:", trnlabfile)

        # بارگذاری فایل متن
        dec_x = np.loadtxt(trnimgfile)  # (N,784)
        dec_y = np.loadtxt(trnlabfile)

        dec_x = dec_x.reshape(dec_x.shape[0], 1, 28, 28)
        print("after reshape dec_x:", dec_x.shape)
        print("labels distribution:", collections.Counter(dec_y))

        resx, resy = [], []
        for cls_ in range(1,10):
            xclass = dec_x[dec_y==cls_]
            yclass = dec_y[dec_y==cls_]
            print(f"class {cls_} shape:", xclass.shape)

            if len(xclass)==0:
                print("  -> No samples in this fold for class", cls_)
                continue

            xclass_t = torch.Tensor(xclass).to(device)
            zclass_t = encoder(xclass_t)
            zclass_np = zclass_t.detach().cpu().numpy()

            needed = imbal[0] - imbal[cls_]
            if needed<=0:
                print(f"Skip class {cls_}, no additional samples.")
                continue

            # SMOTE
            z_synth, y_synth = G_SM1(zclass_np, yclass, needed, cls_)
            z_synth_t = torch.Tensor(z_synth).to(device)
            x_synth_t = decoder(z_synth_t)
            x_synth_np = x_synth_t.detach().cpu().numpy()

            resx.append(x_synth_np)
            resy.append(np.array(y_synth))

        if len(resx)>0:
            syn_x = np.vstack(resx)
            syn_y = np.hstack(resy)
            print("All synthetic shape:", syn_x.shape, syn_y.shape)
        else:
            syn_x = np.zeros((0,1,28,28))
            syn_y = np.array([])

        # ادغام با داده‌های اصلی
        syn_x_flat = syn_x.reshape(syn_x.shape[0], -1)
        dec_x_flat = dec_x.reshape(dec_x.shape[0], -1)
        combx = np.vstack([syn_x_flat, dec_x_flat])
        comby = np.hstack([syn_y, dec_y])

        out_img_file = f"FMNIST/trn_img_f/{m}_trn_img.txt"
        out_lab_file = f"FMNIST/trn_lab_f/{m}_trn_lab.txt"
        np.savetxt(out_img_file, combx)
        np.savetxt(out_lab_file, comby)
        print(f"Saved fold {m} => {out_img_file}, {out_lab_file}")

    t1 = time.time()
    print(f"Done. total time(min): {(t1 - t0)/60:.2f}")

if __name__=='__main__':
    main()
