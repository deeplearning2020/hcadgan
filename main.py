from __future__ import print_function
import argparse
import os
import random
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from DropBlock_attention import DropBlock2D
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--batchSize', type=int, default=200)
parser.add_argument('--imageSize', type=int, default=28)
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--netG', default='')
parser.add_argument('--netD', default='')
parser.add_argument('--manualSeed', type=int)
parser.add_argument('--decreasing_lr', default='120,240,420,620,800')
parser.add_argument('--wd', type=float, default=0.001)
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)

opt = parser.parse_args()
opt.outf = 'model'
opt.cuda = torch.cuda.is_available()

if opt.cuda:
    torch.cuda.set_device(0)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = False

def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

def padWithZeros(X, margin=5):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    newX[margin:X.shape[0] + margin, margin:X.shape[1] + margin, :] = X
    return newX

def createImageCubes(X, y, windowSize=11, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = []
    patchesLabels = []
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            label = y[r-margin, c-margin]
            if label > 0 or not removeZeroLabels:
                patchesData.append(patch)
                patchesLabels.append(label)
    return np.array(patchesData), np.array(patchesLabels)

def load_data():
    data_dir = os.path.join(os.getcwd(), 'data')
    matfn1 = os.path.join(data_dir, 'Indian_pines_corrected.mat')
    data1 = sio.loadmat(matfn1)
    X = data1['indian_pines_corrected']
    matfn2 = os.path.join(data_dir, 'Indian_pines_gt.mat')
    data2 = sio.loadmat(matfn2)
    y = data2['indian_pines_gt']
    return X, y

def prepare_data(X, y, pca_components=3):
    X_pca = applyPCA(X, numComponents=pca_components)
    patches, labels = createImageCubes(X_pca, y, windowSize=11)
    labels = labels - 1
    
    total_samples = len(patches)
    train_size = min(2000, int(0.7 * total_samples))
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    Xtrain = patches[train_indices].transpose(0, 3, 1, 2).astype('float32')
    ytrain = labels[train_indices]
    Xtest = patches[test_indices].transpose(0, 3, 1, 2).astype('float32')
    ytest = labels[test_indices]
    
    return Xtrain, ytrain, Xtest, ytest

class TrainDS(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.len = x_data.shape[0]
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.LongTensor(y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class TestDS(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.len = x_data.shape[0]
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.LongTensor(y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

class netG(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(netG, self).__init__()
        self.ReLU = nn.LeakyReLU(0.2, inplace=True)
        self.Tanh = nn.Tanh()
        self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 2, 1, 0, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(ngf * 8)
        self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 1, 0, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ngf * 4)
        self.Drop2 = DropBlock2D()
        self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 1, 0, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ngf * 2)
        self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf, 3, 1, 0, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(ngf)
        self.conv5 = nn.ConvTranspose2d(ngf, nc, 4, 1, 0, bias=False)
        self.apply(weights_init)

    def forward(self, input):
        x = self.conv1(input)
        x = self.BatchNorm1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.ReLU(x)
        x = self.Drop2(x)
        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.ReLU(x)
        x = self.conv4(x)
        x = self.BatchNorm4(x)
        x = self.ReLU(x)
        x = self.conv5(x)
        output = self.Tanh(x)
        return output

class netD(nn.Module):
    def __init__(self, ndf, nc, nb_label):
        super(netD, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.conv1 = nn.Conv2d(nc, ndf, 3, 1, 0, bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(ndf)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, 3, 1, 0, bias=False)
        self.BatchNorm2 = nn.BatchNorm2d(ndf * 2)
        self.Drop2 = DropBlock2D()
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 0, bias=False)
        self.BatchNorm3 = nn.BatchNorm2d(ndf * 4)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 0, bias=False)
        self.BatchNorm4 = nn.BatchNorm2d(ndf * 8)
        self.conv5 = nn.Conv2d(ndf * 8, ndf * 2, 3, 1, 0, bias=False)
        self.aux_linear = nn.Linear(ndf * 2, nb_label + 1)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.ndf = ndf
        self.apply(weights_init)

    def forward(self, input):
        x = self.conv1(input)
        x = self.LeakyReLU(x)
        x = self.conv2(x)
        x = self.BatchNorm2(x)
        x = self.LeakyReLU(x)
        x = self.Drop2(x)
        x = self.conv3(x)
        x = self.BatchNorm3(x)
        x = self.LeakyReLU(x)
        x = self.conv4(x)
        x = self.BatchNorm4(x)
        x = self.LeakyReLU(x)
        x = self.conv5(x)
        x = x.view(-1, self.ndf * 2)
        c = self.aux_linear(x)
        c = self.softmax(c)
        return c

def test(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    return correct, len(labels.data)

def kappa(testData, k):
    dataMat = np.array(testData)
    P0 = np.sum(np.diag(dataMat)) / np.sum(dataMat)
    row_sums = np.sum(dataMat, axis=1)
    col_sums = np.sum(dataMat, axis=0)
    Pe = np.sum(row_sums * col_sums) / (np.sum(dataMat) ** 2)
    cohens_coefficient = (P0 - Pe) / (1 - Pe)
    return float(cohens_coefficient)

def evaluate_model(netD, test_loader, c_criterion, opt, num_class):
    netD.eval()
    test_loss = 0
    right = 0
    all_Label = []
    all_target = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Testing'):
            indx_target = target.clone()
            if opt.cuda:
                data, target = data.cuda(), target.cuda()
            output = netD(data)
            test_loss += c_criterion(output, target).item()
            pred = output.max(1)[1]
            all_Label.extend(pred.cpu().numpy())
            all_target.extend(target.cpu().numpy())
            right += pred.cpu().eq(indx_target).sum()

    test_loss /= len(test_loader)
    acc = 100. * float(right) / len(test_loader.dataset)
    C = confusion_matrix(np.array(all_target), np.array(all_Label))[:num_class, :num_class]
    np.save('c.npy', C)
    k = kappa(C, np.shape(C)[0])
    AA_ACC = np.diag(C) / np.sum(C, 1)
    AA = np.mean(AA_ACC, 0)
    
    return test_loss, acc, AA, k, right

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def main():
    X, y = load_data()
    pca_components = 3
    Xtrain, ytrain, Xtest, ytest = prepare_data(X, y, pca_components)
    
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.workers)

    num_class = len(np.unique(ytrain))
    nz = opt.nz
    ngf = opt.ngf
    ndf = opt.ndf
    nc = pca_components
    nb_label = num_class

    netG = netG(nz, ngf, nc)
    netD = netD(ndf, nc, nb_label)

    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))

    s_criterion = nn.BCELoss()
    c_criterion = nn.NLLLoss()

    input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    s_label = torch.FloatTensor(opt.batchSize)
    c_label = torch.LongTensor(opt.batchSize)
    f_label = torch.LongTensor(opt.batchSize)

    real_label = 0.8
    fake_label = 0.2

    if opt.cuda:
        netD.cuda()
        netG.cuda()
        s_criterion.cuda()
        c_criterion.cuda()
        input, s_label = input.cuda(), s_label.cuda()
        c_label = c_label.cuda()
        f_label = f_label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    input = Variable(input)
    s_label = Variable(s_label)
    c_label = Variable(c_label)
    f_label = Variable(f_label)
    noise = Variable(noise)
    fixed_noise = Variable(fixed_noise)

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=0.02)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.9, 0.999), weight_decay=0.005)

    decreasing_lr = list(map(int, opt.decreasing_lr.split(',')))
    best_acc = 0

    for epoch in range(1, opt.niter + 1):
        netD.train()
        netG.train()
        right = 0
        if epoch in decreasing_lr:
            optimizerD.param_groups[0]['lr'] *= 0.9
            optimizerG.param_groups[0]['lr'] *= 0.9

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{opt.niter}')
        for i, datas in enumerate(train_pbar):
            for j in range(2):
                netD.zero_grad()
                img, label = datas
                batch_size = img.size(0)
                input.resize_(img.size()).copy_(img)
                s_label.resize_(batch_size).fill_(real_label)
                c_label.resize_(batch_size).copy_(label)
                c_output = netD(input)

                c_errD_real = c_criterion(c_output, c_label)
                errD_real = c_errD_real
                errD_real.backward()
                D_x = c_output.data.mean()

                correct, length = test(c_output, c_label)

                noise.resize_(batch_size, nz, 1, 1)
                noise.normal_(0, 1)
                noise_ = np.random.normal(0, 1, (batch_size, nz, 1, 1))
                noise.resize_(batch_size, nz, 1, 1).copy_(torch.from_numpy(noise_))

                label = np.full(batch_size, nb_label)
                f_label.data.resize_(batch_size).copy_(torch.from_numpy(label))

                fake = netG(noise)
                c_output = netD(fake.detach())
                c_errD_fake = c_criterion(c_output, f_label)
                errD_fake = c_errD_fake
                errD_fake.backward()
                D_G_z1 = c_output.data.mean()
                errD = errD_real + errD_fake
                optimizerD.step()

            netG.zero_grad()
            c_output = netD(fake)
            c_errG = c_criterion(c_output, c_label)
            errG = c_errG
            errG.backward()
            D_G_z2 = c_output.data.mean()
            optimizerG.step()
            right += correct

            train_pbar.set_postfix({
                'D(x)': f'{D_x:.4f}',
                'D(G(z))': f'{D_G_z2:.4f}',
                'Loss_D': f'{errD.item():.4f}',
                'Loss_G': f'{errG.item():.4f}'
            })

        if epoch % 5 == 0:
            train_acc = 100. * right / len(train_loader.dataset)
            print(f"\nTrain Accuracy: {train_acc:.2f}%")
            
            test_loss, acc, AA, k, right = evaluate_model(netD, test_loader, c_criterion, opt, num_class)
            
            print(f"Test Results:")
            print(f"- Average loss: {test_loss:.4f}")
            print(f"- Accuracy: {right}/{len(test_loader.dataset)} ({acc:.2f}%)")
            print(f"- Average Accuracy: {AA:.2f}%")
            print(f"- Kappa Coefficient: {k:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                print(f"New best accuracy: {best_acc:.2f}%")
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict_G': netG.state_dict(),
                    'state_dict_D': netD.state_dict(),
                    'best_acc': best_acc,
                    'optimizer_G': optimizerG.state_dict(),
                    'optimizer_D': optimizerD.state_dict(),
                }, filename=os.path.join(opt.outf, f'best_model.pth'))

if __name__ == '__main__':
    main()
