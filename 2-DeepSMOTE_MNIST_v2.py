import collections
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import os

# برای نمایش برخی نمونه‌های بازسازی‌شده
import matplotlib.pyplot as plt

# برای متریک‌های طبقه‌بندی
from sklearn.metrics import classification_report, accuracy_score

print(torch.version.cuda) #10.1
t3 = time.time()

"""args for AE"""
args = {}
args['dim_h'] = 64         # ابعاد لایه‌های پنهان
args['n_channel'] = 1      # تعداد کانال: MNIST سیاه‌وسفید=1
args['n_z'] = 300          # ابعاد فضای نهان
args['sigma'] = 1.0        # واریانس فضای نهان (در این کد کمتر استفاده شده)
args['lambda'] = 0.01      # پارامتر وزن ضرر یا بخش‌های دیگر
args['lr'] = 0.0002        # نرخ یادگیری
args['epochs'] = 20       # تعداد تکرارها (اپوک برای آموزش AE)
args['batch_size'] = 100   
args['save'] = True       
args['train'] = True      
args['dataset'] = 'mnist' 

##############################################################################
"""مدل Encoder و Decoder"""

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']

        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h, self.dim_h*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h*2, self.dim_h*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h*4, self.dim_h*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h*8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # لایه‌ی کاملاً متصل نهایی برای رسیدن به بردار نهان
        self.fc = nn.Linear(self.dim_h*(2**3), self.n_z)

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze()  # تبدیل به بردار تخت
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
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h*8, self.dim_h*4, 4),
            nn.BatchNorm2d(self.dim_h*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h*4, self.dim_h*2, 4),
            nn.BatchNorm2d(self.dim_h*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h*2, 1, 4, stride=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.dim_h*8, 7, 7)
        x = self.deconv(x)
        return x

"""توابع کنترلی پارامترها"""


def normalize_to_minus1_plus1(x):
    # x فرض می‌شود که در بازه [0,255] باشد
    x = x / 255.0
    x = x * 2.0 - 1.0
    return x

def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True

def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False

"""تابع Borderline-SMOTE (برای جایگزینی SMOTE کلاسیک)"""

def borderline_smote(
    X, y,
    minority_class,
    n_to_sample=1000,
    n_neighbors=5,
    borderline_threshold=0.5,
    random_state=None
):
    if random_state is not None:
        np.random.seed(random_state)

    # جدا کردن اقلیت و اکثریت
    X_min = X[y == minority_class]
    X_maj = X[y != minority_class]

    n_min, n_feat = X_min.shape

    # ساخت KNN
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(X)
    distances, indices = neigh.kneighbors(X_min)

    # شناسایی نقاط مرزی
    borderline_indices = []
    for i in range(n_min):
        neighbors_idx = indices[i]
        neighbors_labels = y[neighbors_idx]
        n_major_neighbors = np.sum(neighbors_labels != minority_class)
        if n_major_neighbors / n_neighbors >= borderline_threshold:
            borderline_indices.append(i)

    if len(borderline_indices) == 0:
        # اگر نقطه مرزی نداریم، کل نقاط اقلیت را در نظر می‌گیریم
        borderline_indices = list(range(n_min))

    # انتخاب تصادفی از میان نقاط مرزی
    base_indices = np.random.choice(borderline_indices, size=n_to_sample, replace=True)

    # ساخت KNN فقط بین اقلیت‌ها
    nn_min = NearestNeighbors(n_neighbors=n_neighbors)
    nn_min.fit(X_min)
    _, min_indices = nn_min.kneighbors(X_min)

    samples = []
    for idx in base_indices:
        neighbors_of_idx = min_indices[idx]
        # یکی از همسایه‌ها را تصادفی انتخاب می‌کنیم
        neighbor_idx = np.random.choice(neighbors_of_idx[1:])
        lam = np.random.rand()
        sample = X_min[idx] + lam * (X_min[neighbor_idx] - X_min[idx])
        samples.append(sample)

    samples = np.array(samples)
    y_samples = np.array([minority_class] * n_to_sample)

    X_new = np.vstack([X, samples])
    y_new = np.hstack([y, y_samples])

    return X_new, y_new


base_dir = 'MNIST'
dtrnimg = os.path.join(base_dir, 'trn_img_f')
dtrnlab = os.path.join(base_dir, 'trn_lab_f')
models_dir = os.path.join(base_dir, 'models', 'crs5')

ids = os.listdir(dtrnimg)
idtri_f = [os.path.join(dtrnimg, image_id) for image_id in ids]
print(idtri_f)

ids = os.listdir(dtrnlab)
idtrl_f = [os.path.join(dtrnlab, image_id) for image_id in ids]
print(idtrl_f)


NUM_RUNS = 10  # یا هر تعداد تکرار که می‌خواهید
all_accuracies = []  # ذخیره Accuracyِ میانگین هر تکرار

for run_idx in range(NUM_RUNS):
    print(f"\n\n********** Run {run_idx+1}/{NUM_RUNS} **********\n")
    accs_in_this_run = []  # لیستی برای ثبت Accuracyِ هر فولد در این تکرار

    for i in range(len(ids)):
        print(f"\nFold: {i}")
        encoder = Encoder(args)
        decoder = Decoder(args)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("device:", device)
        decoder = decoder.to(device)
        encoder = encoder.to(device)

        criterion = nn.MSELoss().to(device)
        
        trnimgfile = idtri_f[i]
        trnlabfile = idtrl_f[i]
        
        print("Image file:", trnimgfile)
        print("Label file:", trnlabfile)
        dec_x = np.loadtxt(trnimgfile) 
        dec_y = np.loadtxt(trnlabfile)

        print('train imgs before reshape ',dec_x.shape) 
        print('train labels ',dec_y.shape) 
        print(collections.Counter(dec_y))

         # اعمال نرمال‌سازی: تبدیل مقادیر از [0,255] به [-1,1]
        dec_x = normalize_to_minus1_plus1(dec_x)

        dec_x = dec_x.reshape(dec_x.shape[0],1,28,28)   
        print('train imgs after reshape ',dec_x.shape) 

        # --- مرحله‌ی اعمال Borderline-SMOTE روی کلاس‌های اقلیت ---
        x_2d = dec_x.reshape(dec_x.shape[0], -1)
        y_1d = dec_y

        unique, counts = np.unique(y_1d, return_counts=True)
        for cls_, cnt_ in zip(unique, counts):
            # فرضاً اگر کلاس کمتر از 1000 نمونه دارد، سعی در افزایش آن تا 1000 داریم
            if cnt_ < 1000:
                needed = 1000 - cnt_
                x_2d, y_1d = borderline_smote(
                    x_2d, y_1d,
                    minority_class=cls_,
                    n_to_sample=needed,
                    n_neighbors=5,
                    borderline_threshold=0.5,
                    random_state=42
                )

        dec_x = x_2d.reshape(x_2d.shape[0], 1, 28, 28)
        dec_y = y_1d

        print('After borderline-SMOTE: ', dec_x.shape, dec_y.shape)

        # ساخت DataLoader
        tensor_x = torch.Tensor(dec_x)
        tensor_y = torch.tensor(dec_y, dtype=torch.long)
        dataset_bal = TensorDataset(tensor_x, tensor_y)
        train_loader = DataLoader(dataset_bal, batch_size=args['batch_size'], 
                                  shuffle=True, num_workers=0)

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

                    images = images.to(device)
                    labs = labs.to(device)

                    # فاز Forward
                    z_hat = encoder(images)
                    x_hat = decoder(z_hat)
                    mse = criterion(x_hat, images)  # خطای بازسازی

                    # بخش دوم Penalty (جابجایی نمونه‌های یک کلاس)
                    # انتخاب تصادفی یک کلاس
                    tc = np.random.choice(10, 1)  
                    xbeg = dec_x[dec_y == tc]
                    ybeg = dec_y[dec_y == tc] 
                    xlen = len(xbeg)
                    nsamp = min(xlen, 100)
                    if nsamp < 2:  
                        # اگر کمتر از 2 نمونه داشتیم، از penalty صرف‌نظر می‌کنیم
                        mse2 = 0.0
                    else:
                        ind = np.random.choice(list(range(len(xbeg))), nsamp, replace=False)
                        xclass = xbeg[ind]  
                        xcminus = np.arange(1, nsamp)
                        xcplus = np.append(xcminus, 0)
                        xcnew = xclass[xcplus]

                        # بردن به GPU
                        xcnew_t = torch.Tensor(xcnew).to(device)

                        # encode
                        xclass_t = torch.Tensor(xclass).to(device)
                        xclass_enc = encoder(xclass_t).detach().cpu().numpy()

                        xc_enc = xclass_enc[xcplus]
                        xc_enc_t = torch.Tensor(xc_enc).to(device)
                        # decode
                        ximg = decoder(xc_enc_t)
                        mse2 = criterion(ximg, xcnew_t)

                    if isinstance(mse2, float):
                        comb_loss = mse
                    else:
                        comb_loss = mse + mse2

                    comb_loss.backward()
                    enc_optim.step()
                    dec_optim.step()

                    bs = images.size(0)
                    train_loss += comb_loss.item() * bs
                    tmse_loss += mse.item() * bs
                    if not isinstance(mse2, float):
                        tdiscr_loss += mse2.item() * bs
                
                # میانگین‌گیری
                total_samples = len(train_loader.dataset)
                train_loss /= total_samples
                tmse_loss  /= total_samples
                tdiscr_loss /= total_samples

                print(f"Epoch: {epoch} \tTrain Loss: {train_loss:.6f}\tmse loss: {tmse_loss:.6f}\tmse2 loss: {tdiscr_loss:.6f}")
                
                # ذخیره بهترین مدل
                if train_loss < best_loss:
                    print('Saving..')
                    fold_dir = os.path.join(models_dir, str(i))
                    os.makedirs(fold_dir, exist_ok=True)

                    path_enc = os.path.join(fold_dir, 'bst_enc.pth')
                    path_dec = os.path.join(fold_dir, 'bst_dec.pth')
                    
                    torch.save(encoder.state_dict(), path_enc)
                    torch.save(decoder.state_dict(), path_dec)
                    print(path_enc)
                    print(path_dec)
                    
                    best_loss = train_loss
            
            # ذخیره مدل نهایی
            fold_dir = os.path.join(models_dir, str(i))
            os.makedirs(fold_dir, exist_ok=True)

            path_enc = os.path.join(fold_dir, 'f_enc.pth')
            path_dec = os.path.join(fold_dir, 'f_dec.pth')
            torch.save(encoder.state_dict(), path_enc)
            torch.save(decoder.state_dict(), path_dec)
            print(path_enc)
            print(path_dec)

        t1 = time.time()
        print('total time(min): {:.2f}'.format((t1 - t0)/60))

        # (1) نمایش چند نمونه از خروجی Decoder جهت ارزیابی کیفی
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            sample_indices = np.random.choice(len(dataset_bal), size=8, replace=False)
            sample_images = dataset_bal[:][0][sample_indices]  # shape (8,1,28,28)
            sample_images = sample_images.to(device)
            z_ = encoder(sample_images)
            recon_ = decoder(z_).cpu().numpy()  # shape (8,1,28,28)

        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for idx_ in range(8):
            # تصویر اصلی
            axes[0, idx_].imshow(sample_images[idx_].cpu().squeeze(), cmap='gray')
            axes[0, idx_].set_title("Original")
            axes[0, idx_].axis('off')
            # تصویر بازسازی‌شده
            axes[1, idx_].imshow(recon_[idx_].squeeze(), cmap='gray')
            axes[1, idx_].set_title("Reconstructed")
            axes[1, idx_].axis('off')
        plt.tight_layout()
        plt.show()

        # (2) آموزش یک طبقه‌بند ساده برای محاسبه‌ی متریک‌های طبقه‌بندی
        class SimpleMLP(nn.Module):
            def __init__(self, input_dim=28*28, hidden_dim=256, num_classes=10):
                super(SimpleMLP, self).__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, num_classes)
                )
            def forward(self, x):
                return self.net(x)

        # آماده‌سازی داده‌ی (X,y) برای طبقه‌بندی
        X_cls = dec_x.reshape(dec_x.shape[0], -1)  # (N, 784)
        y_cls = dec_y

        # برای سادگی، همین مجموعه را 80% train , 20% val می‌کنیم
        N = len(X_cls)
        idx_perm = np.random.permutation(N)
        train_size = int(0.8*N)
        train_idx = idx_perm[:train_size]
        val_idx   = idx_perm[train_size:]

        X_train, y_train = X_cls[train_idx], y_cls[train_idx]
        X_val,   y_val   = X_cls[val_idx],   y_cls[val_idx]

        batch_size_cls = 128
        train_data = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
        val_data   = TensorDataset(torch.Tensor(X_val),   torch.LongTensor(y_val))
        train_loader_cls = DataLoader(train_data, batch_size=batch_size_cls, shuffle=True)
        val_loader_cls   = DataLoader(val_data,   batch_size=batch_size_cls, shuffle=False)

        model_cls = SimpleMLP(input_dim=784, hidden_dim=256, num_classes=10).to(device)
        opt_cls = torch.optim.Adam(model_cls.parameters(), lr=1e-3)
        loss_cls = nn.CrossEntropyLoss()

        epochs_cls = 5
        model_cls.train()
        for ep in range(epochs_cls):
            epoch_loss = 0
            for xb, yb in train_loader_cls:
                xb, yb = xb.to(device), yb.to(device)
                opt_cls.zero_grad()
                logits = model_cls(xb)
                L = loss_cls(logits, yb)
                L.backward()
                opt_cls.step()
                epoch_loss += L.item() * xb.size(0)
            epoch_loss /= len(train_loader_cls.dataset)
            print(f"[Classifier] Epoch {ep} | Loss: {epoch_loss:.4f}")

        model_cls.eval()
        all_preds = []
        all_true  = []
        with torch.no_grad():
            for xb, yb in val_loader_cls:
                xb, yb = xb.to(device), yb.to(device)
                out = model_cls(xb)
                preds = out.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_true.append(yb.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_true  = np.concatenate(all_true)

        acc_fold_i = accuracy_score(all_true, all_preds)
        print("Accuracy (fold {0}): {1:.4f}".format(i, acc_fold_i))

        accs_in_this_run.append(acc_fold_i)

    avg_acc_run = np.mean(accs_in_this_run)
    print(f"\n>>> Average Accuracy in run {run_idx+1}: {avg_acc_run:.4f}\n")
    all_accuracies.append(avg_acc_run)

# در پایان همهٔ تکرارها
mean_acc = np.mean(all_accuracies)
std_acc  = np.std(all_accuracies)
print("=============== Final Report ===============")
print(f"Results over {NUM_RUNS} runs:")
print(f"Accuracy = {mean_acc:.4f} ± {std_acc:.4f}")
print("============================================")

t4 = time.time()
print('final time(min): {:.2f}'.format((t4 - t3)/60))
