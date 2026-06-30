import numpy as np
import pandas as pd
import sys
import os
from random import shuffle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# 请确保以下自定义模块存在且路径正确
from models.ginconv import GINConvNet
from utils import *
from create_data import create

# -------------------- 超参数设置 --------------------
TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1500

# -------------------- 训练函数（修改返回平均损失）--------------------
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    total_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        data_mol = data[0].to(device)
        data_frags = data[1].to(device)  # 新增：片段数据
        data_pro = data[2].to(device)

        optimizer.zero_grad()
        output = model(data_mol, data_frags,data_pro)
        loss = loss_fn(output, ((data_mol.y + data_pro.y) / 2).view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(data_mol.x)  # 按样本数累加，便于计算平均

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data_mol.x),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))

    avg_loss = total_loss / len(train_loader.dataset)
    return avg_loss

# -------------------- 预测函数（保持不变）--------------------
def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data_mol = data[0].to(device)
            data_frags = data[1].to(device)  # 新增
            data_pro = data[2].to(device)
            output = model(data_mol, data_frags, data_pro)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, ((data_mol.y + data_pro.y) / 2).view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

# -------------------- 主程序 --------------------
if __name__ == "__main__":
    # 命令行参数：数据集索引（0: kiba, 1: davis），模型选择（此处固定为GINConvNet），GPU编号（可选）
    datasets = [['kiba', 'davis'][int(sys.argv[1])]]
    modeling = GINConvNet
    model_st = modeling.__name__

    cuda_name = "cuda:0"
    if len(sys.argv) > 3:
        cuda_name = "cuda:" + str(int(sys.argv[3]))
    print('cuda_name:', cuda_name)

    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    for dataset in datasets:
        print('\nrunning on ', model_st + '_' + dataset)

        # 加载数据（假设create函数返回train_data, test_data）
        train_data, test_data = create(int(sys.argv[1]))

        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,collate_fn=collate)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,collate_fn=collate)

        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        model_file_name = 'model_' + model_st + '_' + dataset + '.model'
        result_file_name = 'result_' + model_st + '_' + dataset + '.csv'
        log_file_name = 'log_' + model_st + '_' + dataset + '.txt'          # 新增：日志文件

        x_data, y_data = [], []   # 用于绘图

        # 打开日志文件并写入表头
        with open(log_file_name, 'w') as log_file:
            log_file.write('epoch,train_loss,rmse,mse,pearson,spearman,ci,rm2,is_best\n')

            for epoch in range(NUM_EPOCHS):
                # 训练并获取平均训练损失
                train_loss = train(model, device, train_loader, optimizer, epoch + 1)

                # 验证
                G, P = predicting(model, device, test_loader)
                ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P), rm2(G, P)]

                is_best = False
                if ret[1] < best_mse:
                    torch.save(model.state_dict(), model_file_name)
                    with open(result_file_name, 'w') as f:
                        f.write(','.join(map(str, ret)))
                    best_epoch = epoch + 1
                    best_mse = ret[1]
                    best_ci = ret[-2]
                    best_rm2 = ret[-1]
                    is_best = True
                    print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci,best_rm2:', best_mse, best_ci, best_rm2, model_st, dataset)
                else:
                    print(ret[1], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci,best_rm2:', best_mse, best_ci, best_rm2, model_st, dataset)

                # 写入日志文件
                log_line = f"{epoch+1},{train_loss:.6f},{ret[0]:.6f},{ret[1]:.6f},{ret[2]:.6f},{ret[3]:.6f},{ret[4]:.6f},{ret[5]:.6f},{is_best}\n"
                log_file.write(log_line)
                log_file.flush()   # 确保实时写入磁盘

                # 绘图
                y_data.append(ret[1])
                x_data.append(epoch)
                plt.figure(figsize=(10, 6))
                plt.plot(x_data, y_data, 'b*--', alpha=0.5, linewidth=1, label='mse')
                plt.legend()
                plt.xlabel('epoch')
                plt.ylabel('mse')
                plt.savefig('result_' + dataset + '_mse_curve.jpg', dpi=450)
                plt.close()  # 避免内存累积