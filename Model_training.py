import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from Model_utils import *
#from models.DbTrs_ge import DbTrs_ge
#from models.DbTrs_mut import DbTrs_mut
#from models.DbTrs_meth import DbTrs_meth
#from models.DbTrs_ge_meth_mut import DbTrs_ge_meth_mut
from models.MTEGDRP import MTEGDRP
from torch_geometric.loader import DataLoader

import datetime
import argparse
import csv

# training function at each epoch

def train(model, device, train_loader, optimizer, epoch, log_interval, model_st, scaler, use_amp):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    loss_ae = nn.MSELoss()
    avg_loss = []
    weight_fn = 0.01
    weight_ae = 2


    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            if 'VAE' in model_st:
                output, _, decode, log_var, mu = model(data)
                loss = weight_fn*loss_fn(output, data.y.view(-1, 1).float().to(device)) + loss_ae(decode, data.target_mut[:,None,:].float().to(device)) + torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
            elif 'AE' in model_st:
                output, _, decode = model(data)
                loss = weight_fn*loss_fn(output, data.y.view(-1, 1).float().to(device)) + loss_ae(decode, data.target_mut[:,None,:].float().to(device))
            else:
                output, drug_data, mut_data = model(data)
                loss = loss_fn(output, data.y.view(-1, 1).float().to(device))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.target_mut),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

    print(f'Total loss for epoch {epoch}: {sum(avg_loss)/len(avg_loss):.6f}')
    return sum(avg_loss)/len(avg_loss)


def predicting(model, device, loader, model_st, save_prefix=None, split_name='dataset'):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if 'VAE' in model_st:
                    output, _, decode, log_var, mu = model(data)
            elif 'AE' in model_st:
                output, _, decode = model(data)
            else:
                output, drug_data, mut_data = model(data)

            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)

    preds = np.array(total_preds.cpu().detach().numpy().flatten(), dtype=np.float32)
    labels = np.array(total_labels.cpu().detach().numpy().flatten(), dtype=np.float32)

    print(
        '{}: prediction mean={:.6f}, std={:.6f}, unique={}'.format(
            split_name,
            float(np.mean(preds)),
            float(np.std(preds)),
            len(np.unique(preds))
        )
    )

    if save_prefix is not None:
        pd.DataFrame(preds).to_csv(save_prefix + '_preds.csv', index=False)
        pd.DataFrame(labels).to_csv(save_prefix + '_labels.csv', index=False)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def evaluate_predictions(labels, preds):
    """
    计算回归指标。
    当预测值或标签为常数时，不计算Pearson和Spearman。
    """

    label_std = np.std(labels)
    pred_std = np.std(preds)

    if label_std < 1e-12 or pred_std < 1e-12:
        pearson_value = np.nan
        spearman_value = np.nan
    else:
        pearson_value = pearson(labels, preds)
        spearman_value = spearman(labels, preds)

    results = [
        rmse(labels, preds),
        mse(labels, preds),
        pearson_value,
        spearman_value,
        r2(labels, preds),
        mae(labels, preds)
    ]

    return results


def main(modeling, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name, patience, amp):

    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
    for model in modeling:
        model_st = model.__name__
        dataset = 'GDSC'
        train_losses = []
        val_losses = []
        val_pearsons = []
        history = []
        print('\nrunning on ', model_st + '_' + dataset )
        processed_data_file_train = '/home/public_data/jlu/MTEGDRP-main/data/processed/' + dataset + '_train_mix'+'.pt'
        processed_data_file_val = '/home/public_data/jlu/MTEGDRP-main/data/processed/' + dataset + '_val_mix'+'.pt'
        processed_data_file_test = '/home/public_data/jlu/MTEGDRP-main/data/processed/' + dataset + '_test_mix'+'.pt'
        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_val)) or (not os.path.isfile(processed_data_file_test))):
            print('please run Data_encoding.py to prepare data in pytorch format!')
        else:
            train_data = TestbedDataset(root='/home/public_data/jlu/MTEGDRP-main/data', dataset=dataset+'_train_mix')
            val_data = TestbedDataset(root='/home/public_data/jlu/MTEGDRP-main/data', dataset=dataset+'_val_mix')
            test_data = TestbedDataset(root='/home/public_data/jlu/MTEGDRP-main/data', dataset=dataset+'_test_mix')

            train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
            print("CPU/GPU: ", torch.cuda.is_available())

            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            print(device)
            use_amp = bool(amp) and device.type == 'cuda'
            print('Automatic mixed precision: ', use_amp)

            model = model().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            best_mse = 1000
            best_pearson = 1
            best_epoch = -1
            no_improvement = 0
            model_file_name = 'model_MTEGDRP_' + model_st + '_' + dataset +  '.model'
            result_file_name = 'result_MTEGDRP_' + model_st + '_' + dataset +  '.csv'
            history_file_name = 'history_MTEGDRP_' + model_st + '_' + dataset +  '.csv'
            loss_fig_val_name = 'model_MTEGDRP_' + model_st + '_' + dataset + '_loss_val'
            pearson_fig_val_name = 'model_MTEGDRP_' + model_st + '_' + dataset + '_pearson_val'

            model_dir = "/home/public_data/jlu/MTE_new/log/model/"
            result_dir = "/home/public_data/jlu/MTE_new/log/result/"
            evaluation_dir = "/home/public_data/jlu/MTE_new/log/evaluation/"
            pred_dir = "/home/public_data/jlu/MTE_new/data/data_pred/"
            for path in [model_dir, result_dir, evaluation_dir, pred_dir]:
                if not os.path.exists(path):
                    os.makedirs(path)

            for epoch in range(num_epoch):
                train_loss = train(model, device, train_loader, optimizer, epoch+1, log_interval, model_st, scaler, use_amp)
                G, P = predicting(model, device, val_loader, model_st, save_prefix=None, split_name='validation')
                val_result = evaluate_predictions(G, P)

                train_losses.append(train_loss)
                val_losses.append(val_result[1])
                val_pearsons.append(val_result[2])

                scheduler.step(val_result[1])
                current_lr = optimizer.param_groups[0]['lr']

                print(
                    "Epoch{} validation: RMSE={}, MSE={}, Pearson={}, Spearman={}, R2={}, MAE={}".format(
                        epoch+1,
                        val_result[0],
                        val_result[1],
                        val_result[2],
                        val_result[3],
                        val_result[4],
                        val_result[5]
                    )
                )

                history.append([
                    epoch+1,
                    train_loss,
                    val_result[0],
                    val_result[1],
                    val_result[2],
                    val_result[3],
                    val_result[4],
                    val_result[5],
                    current_lr
                ])
                pd.DataFrame(
                    history,
                    columns=[
                        'epoch', 'train_loss', 'val_rmse', 'val_mse',
                        'val_pearson', 'val_spearman', 'val_r2',
                        'val_mae', 'learning_rate'
                    ]
                ).to_csv(result_dir + history_file_name, index=False)

                if val_result[1] < best_mse:
                    torch.save(model.state_dict(), model_dir + model_file_name)

                    best_epoch = epoch+1
                    best_mse = val_result[1]
                    best_pearson = val_result[2]
                    no_improvement = 0

                    print(' rmse improved at epoch ', best_epoch, '; best_mse:', best_mse, model_st, dataset)
                else:
                    no_improvement += 1
                    print(' no improvement since epoch ', best_epoch, '; best_mse, best pearson:', best_mse, best_pearson, model_st, dataset)

                if patience > 0 and no_improvement >= patience:
                    print('Early stopping at epoch ', epoch+1, '; best epoch:', best_epoch)
                    break

            print("Loading best model from epoch:", best_epoch)

            # 加载验证集MSE最低时保存的模型权重
            model.load_state_dict(
                torch.load(model_dir + model_file_name, map_location=device)
            )

            # 最佳模型在测试集上预测一次
            save_prefix = (
                    pred_dir
                    + model_st
                    + "_"
                    + dataset
                    + "_test"
            )

            G_test, P_test = predicting(
                model,
                device,
                test_loader,
                model_st,
                save_prefix=save_prefix,
                split_name='test'
            )

            ret_test = evaluate_predictions(
                G_test,
                P_test
            )

            print(
                "Final test result: "
                "RMSE={}, MSE={}, Pearson={}, "
                "Spearman={}, R2={}, MAE={}".format(
                    ret_test[0],
                    ret_test[1],
                    ret_test[2],
                    ret_test[3],
                    ret_test[4],
                    ret_test[5]
                )
            )

            with open(
                    result_dir
                    + result_file_name,
                    'w'
            ) as f:
                f.write(','.join(map(str, ret_test)))

            draw_loss(train_losses, val_losses, evaluation_dir + loss_fig_val_name)
            draw_pearson(val_pearsons, evaluation_dir + pearson_fig_val_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=0,     help='0: MTEGDRP')
    parser.add_argument('--train_batch', type=int, required=False, default=128,  help='Batch size training set')
    parser.add_argument('--val_batch', type=int, required=False, default=128, help='Batch size validation set')
    parser.add_argument('--test_batch', type=int, required=False, default=128, help='Batch size test set')
    parser.add_argument('--lr', type=float, required=False, default=0.0001, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=100 , help='Number of epoch')
    parser.add_argument('--log_interval', type=int, required=False, default=20, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default="cuda:0", help='Cuda')
    parser.add_argument('--patience', type=int, required=False, default=15, help='Early stopping patience')
    parser.add_argument('--amp', type=int, required=False, default=0, help='1: use automatic mixed precision, 0: do not use')

    args = parser.parse_args()

    MTEGDRP=[MTEGDRP]
    modeling = MTEGDRP[args.model]
    model = [modeling]
    train_batch = args.train_batch
    val_batch = args.val_batch

    test_batch = args.test_batch
    lr = args.lr
    num_epoch = args.num_epoch
    log_interval = args.log_interval
    cuda_name = args.cuda_name
    patience = args.patience
    amp = args.amp

    main(model, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name, patience, amp)
