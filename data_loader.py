import os
import numpy as np

class MNISTFoldLoader:
    def __init__(self, img_dir, lab_dir):
        self.img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir)]
        self.lab_files = [os.path.join(lab_dir, f) for f in os.listdir(lab_dir)]

    def __len__(self):
        return len(self.img_files)

    def load_fold(self, idx):
        x = np.loadtxt(self.img_files[idx])
        y = np.loadtxt(self.lab_files[idx])
        x = x.reshape(-1, 1, 28, 28)
        return x, y
