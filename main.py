import torch
import numpy as np
import argparse
import time
import os

import utils
from holder import Holder

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default="cuda:3")
parser.add_argument('--data', type=str, default="metr", help="data set")
parser.add_argument('--batch_size', type=int, default=38)
parser.add_argument('--in_dim', type=int, default=2)
parser.add_argument('--hid_dim', type=int, default=128)
parser.add_argument('--out_dim', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0005)
parser.add_argument("--clip", type=float, default=5.)
parser.add_argument('--weight_decay', type=float, default=0.000001, help='weight decay rate')
parser.add_argument("--his_len", type=int, default=12, help="")
parser.add_argument("--pred_len", type=int, default=12, help="")
parser.add_argument("--seed", type=int, default=1314, help="random seed")
parser.add_argument('--info_dir', type=str, default="./infos/metr12/ratio003.txt")
parser.add_argument('--channels', type=int, default=2)
parser.add_argument('--layers', type=int, default=5)
parser.add_argument('--snpsts_len', type=int, default=4)
parser.add_argument('--dropout', type=float, default=0.3)

args = parser.parse_args()
utils.record_info('snapshot增加为6，pearson矩阵数量变为0.05', args.info_dir)
utils.record_info(str(args), args.info_dir)
print(args)

if args.data == "metr":
    args.data_path = './data/METR-LA'
    args.adj_mx_path = './data/sensor_graph/adj_mx.pkl'
    args.adj_mx = torch.Tensor(utils.load_pickle(args.adj_mx_path)[-1])
    args.num_node = 207
    args.pearson_path = "./data/METR-LA/pearson_corr.pkl"
    args.dilations = [1, 2, 4, 2, 1, 1]
elif args.data == "bay":
    args.data_path = './data/PEMS-BAY'
    args.adj_mx_path = './data/sensor_graph/adj_mx_bay.pkl'
    args.adj_mx = torch.Tensor(utils.load_pickle(args.adj_mx_path)[-1])
    args.num_node = 325
    args.pearson_path = "./data/PEMS-BAY/pearson_corr.pkl"
    args.dilations = [1, 3, 4, 2, 1]


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataloader = utils.load_dataset(args.data_path, args.batch_size, args.batch_size, args.batch_size)
    args.scaler = dataloader['scaler']

    engine = a(args)
    print("start training...")
    his_loss = []
    val_time = []
    train_time = []
    for epoch_num in range(args.epochs + 1):
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            trainX = torch.Tensor(x).to(args.device)
            trainy = torch.Tensor(y).to(args.device)
            metrics = engine.train(trainX, trainy[:, :, :, 0])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % 500 == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), flush=True)
                utils.record_info(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]), args.info_dir)

        t2 = time.time()
        train_time.append(t2 - t1)
        valid_loss = []
        valid_mape = []
        valid_rmse = []

        print("eval...")
        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            valx = torch.Tensor(x).cuda()
            valy = torch.Tensor(y).cuda()
            metrics = engine.eval(valx, valy[:, :, :, 0])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(epoch_num, (s2 - s1)))
        utils.record_info(log.format(epoch_num, (s2 - s1)), args.info_dir)
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(epoch_num, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse,
                         (t2 - t1)),
              flush=True)
        utils.record_info(
            log.format(epoch_num, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse,
                       (t2 - t1)), args.info_dir)
        torch.save(engine.model, "./model/" + "_epoch_" + str(epoch_num) + ".pkl")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


if __name__ == '__main__':
    main()
