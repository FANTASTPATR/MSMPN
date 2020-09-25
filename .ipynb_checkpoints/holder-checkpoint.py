import torch.optim as optim
import torch

from modelv9 import Model
from optim import Optim
import utils
import torch.optim as op


class Holder():
    def __init__(self, args):
        self.args = args
        self.model = Model(args)
        self.model = self.model.to(self.args.device)
        self.optimizer = op.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.loss = utils.masked_mae
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Total:', total_num, 'Trainable:', trainable_num)

    def train(self, inputs, real):
        self.model.train()
        # self.model.zero_grad()
        # self.optimizer.optimizer.zero_grad()
        self.optimizer.zero_grad()
        output = self.model(inputs)
        real = torch.unsqueeze(real, dim=-1)[:, :self.args.pred_len, :, :]
        predict = self.args.scaler.inv_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward(retain_graph=True)
        if self.args.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.optimizer.step()
        mape = utils.masked_mape(predict, real, 0.0).item()
        rmse = utils.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, input, real_val):
        self.model.eval()
        with torch.no_grad():
            output = self.model(input)
        real = torch.unsqueeze(real_val, dim=-1)[:, :self.args.pred_len, :, :]
        predict = self.args.scaler.inv_transform(output)
        loss = self.loss(predict, real, 0.0)
        mape = utils.masked_mape(predict, real, 0.0).item()
        rmse = utils.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mape, rmse
