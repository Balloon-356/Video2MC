'''
    Author: Guanghan Ning
    E-mail: guanghan.ning@jd.com
    October 24th, 2018

    Processor for Siamese Graph Convolutional Networks for Pose Tracking
'''

# !/usr/bin/env python
# pylint: disable=W0201
import argparse

import numpy as np
# torch
import torch
import torch.optim as optim
# import contrastive loss
from gcn_utils.contrastive import ContrastiveLoss
from gcn_utils.processor_base import Processor
# torchlight
from torchlight import str2bool


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class SGCN_Processor(Processor):
    """
        Processor for Siamese Graph Convolutional Networks (SGCN)
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = ContrastiveLoss(margin=1)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                    0.1 ** np.sum(self.meta_info['epoch'] >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data_1, data_2, label in loader:
            # get data
            data_1 = data_1.float().to(self.dev)
            data_2 = data_2.float().to(self.dev)
            label = label.float().to(self.dev)

            # forward
            feature_1, feature_2 = self.model(data_1, data_2)
            loss = self.loss(feature_1, feature_2, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        result_frag = []
        label_frag = []
        dist_frag = []
        pred_label_frag = []

        for data_1, data_2, label in loader:

            # get data
            data_1 = data_1.float().to(self.dev)
            data_2 = data_2.float().to(self.dev)
            label = label.float().to(self.dev)

            # inference
            with torch.no_grad():
                feature_1, feature_2 = self.model(data_1, data_2)
            result_frag.append(feature_1.data.cpu().numpy())
            result_frag.append(feature_2.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = self.loss(feature_1, feature_2, label)
                loss_value.append(loss.item())
                label_frag.append(label.data.cpu().numpy())

                # euclidian distance
                diff = feature_1 - feature_2
                dist_sq = torch.sum(pow(diff, 2), 1)
                dist = torch.sqrt(dist_sq)

                dist_frag.append(dist.data.cpu().numpy())

                margin = 0.2
                if dist >= margin:
                    pred_label_frag.append(0)  # not match
                else:
                    pred_label_frag.append(1)  # match

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss'] = np.mean(loss_value)
            self.show_epoch_info()

            # print(result_frag)
            # print(label_frag[0:20])
            # print(pred_label_frag[0:20])

            # show accuracy
            accuracy = calculate_accuracy(label_frag, pred_label_frag)
            print("accuracy: {}".format(accuracy))

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser


def calculate_accuracy(label_list, pred_list):
    len_pred = len(pred_list)
    len_label = len(label_list)
    assert (len_pred == len_label)

    num_true = 0
    for id in range(len_pred):
        if label_list[id] == pred_list[id]:
            num_true += 1
    accuracy = num_true * 100.0 / len_pred
    return accuracy
