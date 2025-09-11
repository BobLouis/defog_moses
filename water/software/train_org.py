import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from std_L_ch_based import ResIERes
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image

from PerceptualLoss import LossNetwork
from torchvision.models import vgg16
import numpy as np
import torch.nn.functional as F

class UWDataset(Dataset):
    def __init__(self, samples, img_transform=None):

        self.img_transform = img_transform
        self.samples = samples
        
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        
        img_path = self.samples[idx][0]
        gt_path = self.samples[idx][1]

        img = Image.open(str(img_path)).convert('RGB').resize((256,256))
        gt = Image.open(str(gt_path)).convert('RGB').resize((256,256))
        if self.img_transform:
            img = self.img_transform(img)
            gt = self.img_transform(gt)
        return (img,gt)

if __name__ == "__main__":
    
    BATCH_SIZE = 1
    EPOCHS = 500
    LR = 1e-4
    USE_CUDA = True

    device = 'cpu'
    if USE_CUDA and torch.cuda.is_available():
        device = 'cuda'
    print(f'Using {device} for training.')

    img_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    model_fa = Path(f'UIEB_models/3K_noVar')
    model_fa.mkdir(exist_ok= True)
    for group in range(1,2):


        model_path = Path(model_fa / f'{group}')
        model_path.mkdir(exist_ok= True)

        DATA_PATH = f'./train/train_256'
        files = []
        folder_path = Path(DATA_PATH)
        for i, path in enumerate(folder_path.iterdir()):
            img_path = []
            img_path.append(path.parents[1] / f'train_256' / path.name)
            img_path.append(path.parents[1] / f'train_256_gt' / path.name)
            files.append(img_path)

        dataset = UWDataset(files,img_transforms)

        trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)

        model = ResIERes()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型的總參數量: {total_params}")
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.to(device)
        for param in vgg_model.parameters():
            param.requires_grad = False
        loss_network = LossNetwork(vgg_model).to(device)
        loss_network.eval()


        if device == 'cuda':
            model.cuda()

        
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        
        for epoch in range(0, EPOCHS):
            train_loss_tmp, edge_loss_tmp = [], []
            # Print epoch
            print(f'Starting epoch {epoch+1}')

            # Iterate over the DataLoader for training data
            
            for i, (inputs, targets) in enumerate(trainloader):
                

                inputs, targets = inputs.to(device), targets.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # # Perform forward pass
                model.train()
                outputs = model(inputs)

                # Compute loss
                              
                smooth_loss = F.smooth_l1_loss(outputs, targets)
                perceptual_loss = loss_network(outputs, targets)

                loss = smooth_loss + 0.5*perceptual_loss
                
                # # Perform backward pass
                loss.backward()
                
                # # Perform optimization
                optimizer.step()

                train_loss_tmp.append(loss.item())
                # Print statistics


                if i % 10 == 9:
                    print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, np.mean(train_loss_tmp)))
                        
            # Process is complete.
            print('Training process has finished. Saving trained model.')

            # Print about testing
            print('Starting testing')
            
            # Saving the model
            if epoch % 10 == 9:
                save_path =  model_path / f'model-uw-{epoch}.pth'
                torch.save(model, save_path)
