from trainer import Trainer
from data_loader import MNISTFoldLoader

if __name__ == '__main__':
    args = {
        'dim_h': 64,
        'n_channel': 1,
        'n_z': 300,
        'sigma': 1.0,
        'lambda': 0.01,
        'lr': 0.0002,
        'epochs': 1,
        'batch_size': 100,
        'save': True,
        'train': True
    }
    img_dir = 'MNIST/trn_img/'
    lab_dir = 'MNIST/trn_lab/'
    loader = MNISTFoldLoader(img_dir, lab_dir)
    trainer = Trainer(args, loader)
    trainer.train()