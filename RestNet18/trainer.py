import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, dataset
from torch.utils.tensorboard import SummaryWriter
from net.nets import RestNet18
from net.mobileNetV2 import MobileNetV2
from dataset import Dataset



class Trainer:
    def __init__(self):
        super().__init__()
        self.batch_size = 80
        self.device = 'cuda'
        self.net = RestNet18()
        self.net.to(self.device)
        self.summerWrite = SummaryWriter(log_dir='logs')
        self.train_dataset = Dataset(root=r"D:\AIdata\gutou\arthrosis", is_train=True)
        self.train_loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataset = Dataset(root=r"D:\AIdata\gutou\arthrosis", is_train=False)
        self.test_loader = DataLoader(dataset=self.test_dataset, batch_size=20, shuffle=False)
        self.opt = torch.optim.Adam(self.net.parameters())
        self.fc_loss = nn.CrossEntropyLoss()
        self.save_path = r'params/DIPFirst1.pth'


    def train(self):
        if os.path.exists(self.save_path):
            self.net.load_state_dict(torch.load(self.save_path), strict=False)
            print('加载成功')

        for epoch in range(100):
            sum_loss = 0
            with tqdm(total=int(2851 / self.batch_size), desc=f'epoch {epoch} / 100 ', mininterval=1,
                      ncols=100) as pbar:
                for i, (img, target) in enumerate(self.train_loader):
                    self.net.train()
                    img, target = img.to(self.device), target.to(self.device)
                    h = self.net(img)
                    loss = self.fc_loss(h, target)
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    sum_loss += loss.item()
                    if i % 10 == 0:
                        torch.save(self.net.state_dict(), f'params/DIPFirst1.pth')
                    pbar.update(1)
            avg_loss = sum_loss / len(self.train_loader)
            self.summerWrite.add_scalar('net训练损失', avg_loss, epoch)
            print(f'\n-------轮次: {epoch}--------')
            print(f'net平均损失为：{avg_loss}')
            self.net.load_state_dict(torch.load(r'params//DIPFirst1.pth'))
            sum_score = 0
            for i, (img, target) in enumerate(self.test_loader):
                self.net.eval()
                img, target = img.to(self.device), target.to(self.device)
                h = self.net(img)
                a = torch.argmax(h, dim=1)
                b = torch.argmax(target, dim=1)
                score = torch.mean(torch.eq(a, b).float())
                sum_score += score
            avg_score = sum_score / len(self.test_loader)
            self.summerWrite.add_scalar('测试得分', avg_score, epoch)
            print(f'平均得分: {avg_score}')
            print('-' * 90)
            time.sleep(0.2)

    def test(self):
        for epoch in range(100):
            self.net.load_state_dict(torch.load(r'params//' + os.listdir(r'params')[-1]))
            sum_score = 0
            for i, (img, label) in enumerate(self.test_loader):
                self.net.eval()
                img, label = img.to(self.device), label.to(self.device)
                h = self.net(img)
                a = torch.argmax(h, dim=1)
                b = torch.argmax(label, dim=1)
                score = torch.mean(torch.eq(a, b).float())
                sum_score += score
            avg_score = sum_score / len(self.test_loader)
            self.summerWrite.add_scalar('轮次得分', avg_score, epoch)
            print(f'轮次：{epoch}\t 平均得分：{avg_score}')

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()