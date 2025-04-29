import os
import time
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from encoder import Encoder
from decoder import Decoder
from smote_generator import SMOTEGenerator
from data_loader import MNISTFoldLoader

class Trainer:
    def __init__(self, args, data_loader):
        self.args = args
        self.loader = data_loader
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self):
        for i in range(len(self.loader)):
            print(f"Fold: {i}")
            enc = Encoder(self.args).to(self.device)
            dec = Decoder(self.args).to(self.device)
            criterion = nn.MSELoss().to(self.device)
            enc_opt = torch.optim.Adam(enc.parameters(), lr=self.args['lr'])
            dec_opt = torch.optim.Adam(dec.parameters(), lr=self.args['lr'])
            best_loss = np.inf
            x_raw, y_raw = self.loader.load_fold(i)
            print('Data shapes:', x_raw.shape, y_raw.shape, collections.Counter(y_raw))
            tensor_x = torch.Tensor(x_raw)
            tensor_y = torch.tensor(y_raw, dtype=torch.long)
            dataset = TensorDataset(tensor_x, tensor_y)
            loader = DataLoader(dataset, batch_size=self.args['batch_size'], shuffle=True)

            start = time.time()
            for ep in range(self.args['epochs']):
                train_loss = 0.0
                for images, labs in loader:
                    enc.zero_grad()
                    dec.zero_grad()
                    images = images.to(self.device)
                    z = enc(images)
                    x_hat = dec(z)
                    mse = criterion(x_hat, images)

                    tc = np.random.choice(10)
                    x_c, y_c = SMOTEGenerator.biased_get_class(x_raw, y_raw, tc)
                    nsamp = min(len(x_c), self.args['batch_size'])
                    idx = np.random.choice(len(x_c), nsamp, replace=False)
                    x_sel = torch.Tensor(x_c[idx]).to(self.device)
                    x_sel_next = torch.Tensor(x_c[np.append(np.arange(1, nsamp), 0)]).to(self.device)

                    enc_out = enc(x_sel).detach().cpu().numpy()
                    enc_sel = torch.Tensor(enc_out[np.append(np.arange(1, nsamp), 0)]).to(self.device)
                    x_img = dec(enc_sel)
                    mse2 = criterion(x_img, x_sel_next)

                    loss = mse + mse2
                    loss.backward()
                    enc_opt.step()
                    dec_opt.step()
                    train_loss += loss.item() * images.size(0)

                avg_loss = train_loss / len(loader)
                print(f"Epoch {ep} | Loss: {avg_loss:.6f}")
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self._save_model(enc, dec, i, 'bst')
            self._save_model(enc, dec, i, 'final')
            print(f"Fold {i} time (min): {(time.time() - start)/60:.2f}\n")

    def _save_model(self, enc, dec, fold, tag):
        base = f"MNIST/models/crs5/{fold}"
        os.makedirs(base, exist_ok=True)
        torch.save(enc.state_dict(), os.path.join(base, f"{tag}_enc.pth"))
        torch.save(dec.state_dict(), os.path.join(base, f"{tag}_dec.pth"))