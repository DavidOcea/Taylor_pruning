import torch.nn as nn
from .import backbones
# from ext_layers import ArcFullyConnected

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import random
import math

def where(cond, x_1, x_2):
    cond = cond.float()
    return (cond * x_1) + ((1-cond) * x_2)

class ArcFullyConnected(Module):

    def __init__(self, in_features, out_features, s, m, is_pw=True, is_hard=False):
        super(ArcFullyConnected, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.is_pw = is_pw
        self.is_hard = is_hard
        assert s > 0
        assert 0 <= m < 0.5* math.pi
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def __repr__(self):
        return ('in_features={}, out_features={}, s={}, m={}'
                .format(self.in_features, self.out_features, self.s, self.m))

    def forward(self, embed, label):
        n_weight = F.normalize(self.weight, p=2, dim=1)
        n_embed = F.normalize(embed, p=2, dim=1)*self.s
        out = F.linear(n_embed, n_weight)
        score = out.gather(1, label.view(-1, 1))
        cos_y = score / self.s
        sin_y = torch.sqrt(1 - cos_y**2)
        arc_score = self.s * (cos_y*math.cos(self.m) - sin_y*math.sin(self.m))
        if self.is_pw:
            if not self.is_hard:
                arc_score = where(score > 0, arc_score, score)
            else:
                mm = math.sin(math.pi - self.m)*self.m # actually it is sin(m)*m
                th = math.cos(math.pi - self.m) # actually it is -cos(m)
                arc_score = where((score-th) > 0, arc_score, score-self.s*mm)
        one_hot = Variable(torch.cuda.FloatTensor(out.shape).fill_(0))
        out += (arc_score - score) * one_hot.scatter_(1, label.view(-1, 1), 1)
        return out



class MultiTaskWithLoss(nn.Module):
    def __init__(self, backbone, num_classes, feature_dim, spatial_size, arc_fc=False, feat_bn=False, s=64, m=0.5, is_pw=True, is_hard=False):
        super(MultiTaskWithLoss, self).__init__()
        self.feat_bn = feat_bn
        self.basemodel = backbones.__dict__[backbone](num_classes=num_classes)
        if feat_bn:
            self.bn1d = nn.BatchNorm1d(feature_dim, affine=False, eps=2e-5, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        if num_classes is not None:
            self.num_tasks = len(num_classes)
            self.arc_fc = arc_fc
            if not arc_fc:
                self.fcs = nn.ModuleList([nn.Linear(feature_dim, num_classes[k]) for k in range(self.num_tasks)])
            else:
                self.fcs = nn.ModuleList([ArcFullyConnected(feature_dim, num_classes[k], s, m, is_pw, is_hard) for k in range(self.num_tasks)])


    def forward(self,input,target=None,slice_idx=[0, 128, 256, 384],eval_mode=False,mixup_criterion=None,mixup_data=None, mix_alpha=None):
        if mix_alpha is not None:
            inputs = [input[slice_idx[k]:slice_idx[k+1]] for k in range(self.num_tasks)]
            target_slice = [target[slice_idx[k]:slice_idx[k+1]] for k in range(self.num_tasks)]
            mix_data = [mixup_data(ins,tag,mix_alpha, True) for ins,tag in zip(inputs,target_slice)]
            m_feture = [self.basemodel(ins[0]) for ins in mix_data]
            y_p = [self.fcs[k](m_feture[k]) for k in range(self.num_tasks)]
            return [mixup_criterion(self.criterion,xx,md[1],md[2],md[3]) for xx, md in zip(y_p, mix_data)]
        
        feature = self.basemodel(input)
        if eval_mode:
            x = [self.fcs[k](feature) for k in range(self.num_tasks)]
            return x
        else:
            if not self.arc_fc:
                x = [self.fcs[k](feature[slice_idx[k]:slice_idx[k+1], ...]) for k in range(self.num_tasks)]
            else:
                x = [self.fcs[k](feature[slice_idx[k]:slice_idx[k+1], ...],
                    target[slice_idx[k]:slice_idx[k+1]]) for k in range(self.num_tasks)]
            target_slice = [target[slice_idx[k]:slice_idx[k+1]] for k in range(self.num_tasks)]
            return [self.criterion(xx, tg) for xx, tg in zip(x, target_slice)]




if __name__ == '__main__':
    model = MultiTaskWithLoss(backbone='resnet18', num_classes=[3], feature_dim=512, spatial_size=112, arc_fc=False, feat_bn=False)
    # for name, param in model.named_parameters():
	#     print(name, '      ', param.size())
    print(model)
    # summary(model, (3, 112, 112))
#     hl.build_graph(model, torch.ones([1, 3, 112, 112]))