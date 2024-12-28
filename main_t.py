import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import scipy.io as sio
import os
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from datetime import datetime

def apply_pca(X, num_components):
    X_reshaped = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    X_transformed = pca.fit_transform(X_reshaped)
    return np.reshape(X_transformed, (X.shape[0], X.shape[1], num_components))

def pad_with_zeros(X, margin=5):
    new_X = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    new_X[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return new_X

def create_patches(X, y, patch_size=11):
    margin = patch_size // 2
    padded_X = pad_with_zeros(X, margin=margin)
    patches_data = []
    patches_labels = []
    
    for r in range(margin, padded_X.shape[0] - margin):
        for c in range(margin, padded_X.shape[1] - margin):
            if y[r-margin, c-margin] > 0:
                patch = padded_X[r-margin:r+margin+1, c-margin:c+margin+1, :]
                patches_data.append(patch)
                patches_labels.append(y[r-margin, c-margin] - 1)
    
    return np.array(patches_data), np.array(patches_labels)

def save_model(netG, netD, optimizerG, optimizerD, epoch, metrics, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    model_info = {
        'epoch': epoch,
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'test_accuracy': metrics['test_acc'],
        'average_accuracy': metrics['aa'],
        'kappa': metrics['kappa'],
        'train_accuracy': metrics['train_acc']
    }
    
    model_path = os.path.join(save_dir, f'best_model_{timestamp}.pth')
    torch.save(model_info, model_path)
    print(f"\nBest model saved at: {model_path}")
    print(f"Test Accuracy: {metrics['test_acc']:.2f}%")
    print(f"Average Accuracy: {metrics['aa']:.4f}")
    print(f"Kappa: {metrics['kappa']:.4f}")
    
    metrics_file = os.path.join(save_dir, f'metrics_{timestamp}.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Test Accuracy: {metrics['test_acc']:.2f}%\n")
        f.write(f"Average Accuracy: {metrics['aa']:.4f}\n")
        f.write(f"Kappa: {metrics['kappa']:.4f}\n")
        f.write(f"Train Accuracy: {metrics['train_acc']:.2f}%\n")

class HSIDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return len(self.y)

class Discriminator(nn.Module):
    def __init__(self, ndf, nc, num_classes):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(nc, ndf, 3, 1, 1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ndf * 8, num_classes + 1)
        self.dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 3, 2, 1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf, nc, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.main(x)

def train_epoch(netD, netG, train_loader, optimizerD, optimizerG, criterion, device, nz):
    netD.train()
    netG.train()
    total_correct = 0
    total_samples = 0
    
    for data, target in tqdm(train_loader, desc="Training"):
        batch_size = data.size(0)
        data = data * 2 - 1  
        data, target = data.to(device), target.to(device)
        
        netD.zero_grad()
        output_real = netD(data)
        errD_real = criterion(output_real, target)
        errD_real.backward()
        
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        fake_labels = torch.full((batch_size,), output_real.size(1)-1, device=device, dtype=torch.long)
        output_fake = netD(fake.detach())
        errD_fake = criterion(output_fake, fake_labels)
        errD_fake.backward()
        optimizerD.step()
        
        netG.zero_grad()
        output_fake = netD(fake)
        errG = criterion(output_fake, target)
        errG.backward()
        optimizerG.step()
        
        pred = output_real.max(1)[1]
        total_correct += pred.eq(target).sum().item()
        total_samples += batch_size
    
    return 100. * total_correct / total_samples

def evaluate(netD, test_loader, criterion, device):
    netD.eval()
    test_loss = 0
    correct = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data = data * 2 - 1  
            data, target = data.to(device), target.to(device)
            output = netD(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1)[1]
            correct += pred.eq(target).sum().item()
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy, all_preds, all_targets

def calculate_metrics(true_labels, pred_labels, n_classes):
    cm = confusion_matrix(true_labels, pred_labels)[:n_classes, :n_classes]
    per_class_acc = []
    for i in range(n_classes):
        if np.sum(cm[i, :]) != 0:
            per_class_acc.append(cm[i, i] / np.sum(cm[i, :]))
    aa = np.mean(per_class_acc) if per_class_acc else 0
    
    total = np.sum(cm)
    if total > 0:
        pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / (total ** 2)
        po = np.sum(np.diag(cm)) / total
        kappa = (po - pe) / (1 - pe) if pe != 1 else 0
    else:
        kappa = 0
        
    return aa, kappa

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_path = os.path.join(os.getcwd(), 'data')
    X = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
    y = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    
    n_components = 3
    patch_size = 11
    batch_size = 64
    nz = 100
    ngf = 64
    ndf = 64
    num_epochs = 500
    lr = 0.0001
    
    print("Applying PCA...")
    X_pca = apply_pca(X, n_components)
    print("Creating patches...")
    patches_data, patches_labels = create_patches(X_pca, y, patch_size)
    
    n_samples = len(patches_labels)
    n_train = 2000
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train = patches_data[train_indices].transpose(0, 3, 1, 2)
    y_train = patches_labels[train_indices]
    X_test = patches_data[test_indices].transpose(0, 3, 1, 2)
    y_test = patches_labels[test_indices]
    
    train_dataset = HSIDataset(X_train, y_train)
    test_dataset = HSIDataset(X_test, y_test)
    
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    num_classes = len(np.unique(y)) - 1
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    netG = Generator(nz, ngf, n_components).to(device)
    netD = Discriminator(ndf, n_components, num_classes).to(device)
    
    criterion = nn.NLLLoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    
    scheduler_D = optim.lr_scheduler.StepLR(optimizerD, step_size=100, gamma=0.5)
    scheduler_G = optim.lr_scheduler.StepLR(optimizerG, step_size=100, gamma=0.5)
    
    best_acc = 0
    
    for epoch in range(num_epochs):
        train_acc = train_epoch(netD, netG, train_loader, optimizerD, optimizerG, criterion, device, nz)
        scheduler_D.step()
        scheduler_G.step()
        
        if epoch % 5 == 0:
            test_loss, test_acc, all_preds, all_targets = evaluate(netD, test_loader, criterion, device)
            aa, kappa = calculate_metrics(all_targets, all_preds, num_classes)
            
            print(f'Epoch: {epoch}')
            print(f'Train Accuracy: {train_acc:.2f}%')
            print(f'Test Accuracy: {test_acc:.2f}%')
            print(f'Average Accuracy: {aa:.4f}')
            print(f'Kappa: {kappa:.4f}')
            
            if test_acc > best_acc:
                best_acc = test_acc
                metrics = {
                    'test_acc': test_acc,
                    'aa': aa,
                    'kappa': kappa,
                    'train_acc': train_acc
                }
                save_model(netG, netD, optimizerG, optimizerD, epoch, metrics)

if __name__ == '__main__':
    main()
