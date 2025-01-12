from torch import nn
import torch
import torch.nn.functional as F
from core import resnet
import numpy as np
from core.anchors import generate_default_anchor_maps, hard_nms
from config import CAT_NUM, PROPOSAL_NUM


class ProposalNet(nn.Module):
    def __init__(self):
        super(ProposalNet, self).__init__()
        self.down1 = nn.Conv2d(2048, 128, 3, 1, 1)
        self.down2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.ReLU = nn.ReLU()
        self.tidy1 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy2 = nn.Conv2d(128, 6, 1, 1, 0)
        self.tidy3 = nn.Conv2d(128, 9, 1, 1, 0)

    def forward(self, x):
        batch_size = x.size(0)
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1).view(batch_size, -1)
        t2 = self.tidy2(d2).view(batch_size, -1)
        t3 = self.tidy3(d3).view(batch_size, -1)
        return torch.cat((t1, t2, t3), dim=1)  # 横着cat三次卷积清理后的结果


class attention_net(nn.Module):
    def __init__(self, topN=4):
        super(attention_net, self).__init__()
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, 200)
        self.proposal_net = ProposalNet()
        self.topN = topN
        self.concat_net = nn.Linear(2048 * (CAT_NUM + 1), 200)
        self.partcls_net = nn.Linear(512 * 4, 200)
        _, edge_anchors, _ = generate_default_anchor_maps()
        self.pad_side = 224
        self.edge_anchors = (edge_anchors + 224).astype(np.int)  # 为什么要加上（右移）224，这样不好吧？

    def forward(self, x):  # x: (448, 448)
        # 图片x送入resnet得到输出resnet_out和抽取到的两个特征：rpn_feature,、feature
        resnet_out, rpn_feature, feature = self.pretrained_model(x)  # resnet50的标准输入不是224×224吗？

        # 对图片x四周0填充
        x_pad = F.pad(x, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        batch = x.size(0)

        # rpn_feature通过proposal_net得到rpn_score。rpn_score: (batch * nb_anchor), nb_anchor为一张图片产生的一类（如edge anchors）anchors总个数 = ceil(w/stride) × ceil(h/stride)
        rpn_score = self.proposal_net(rpn_feature.detach())

        # all_cdds：3维，第0维batch_size，第1、2维：每张图片的所有cdd：第一列：每个anchor的得分，第二、三、四、五列：每个anchor的位置，第六列：anchor的索引
        all_cdds = [
            np.concatenate((x.reshape(-1, 1), self.edge_anchors.copy(), np.arange(0, len(x)).reshape(-1, 1)), axis=1)
            for x in rpn_score.data.cpu().numpy()]

        # 选出topN个cdd。top_n_cdds：3维，第0维batch_size，第1、2维：每张图片的topN张cdd：第一列：得分，第二、三、四、五列：topN个anchor的位置，第六列：索引
        top_n_cdds = [hard_nms(x, topn=self.topN, iou_thresh=0.25) for x in all_cdds]
        top_n_cdds = np.array(top_n_cdds)

        # 取出最后一列anchor的索引：top_n_index：(batch_size, topN)
        top_n_index = top_n_cdds[:, :, -1].astype(np.int)
        top_n_index = torch.from_numpy(top_n_index).cuda()
        # 得到n张图片的得分：top_n_prob：(batch_size, topN)
        top_n_prob = torch.gather(rpn_score, dim=1, index=top_n_index)

        # 得到所有anchor指定的图片，并统一到224×224
        part_imgs = torch.zeros([batch, self.topN, 3, 224, 224]).cuda()
        for i in range(batch):
            for j in range(self.topN):
                [y0, x0, y1, x1] = top_n_cdds[i][j, 1:5].astype(np.int)
                part_imgs[i:i + 1, j] = F.interpolate(x_pad[i:i + 1, :, y0:y1, x0:x1], size=(224, 224), mode='bilinear',
                                                      align_corners=True)  # 选取anchor，再上采样到224×224
        part_imgs = part_imgs.view(batch * self.topN, 3, 224, 224)

        # 提取part_imgs的特征
        _, _, part_features = self.pretrained_model(part_imgs.detach())
        part_feature = part_features.view(batch, self.topN, -1)

        # 取前CAT_NUM个特征，和整图提取的特征concat起来，得到concat_out
        part_feature = part_feature[:, :CAT_NUM, ...].contiguous()
        part_feature = part_feature.view(batch, -1)
        concat_out = torch.cat([part_feature, feature], dim=1)

        # concat_out的全连接分类层结果raw_logits：batch_size × 200
        concat_logits = self.concat_net(concat_out)  # 全连接。怎么保证的concat_out的第一维长度是2048*(CAT_NUM+1)
        # 整图的全连接分类层结果raw_logits：batch_size × ？ × 200
        raw_logits = resnet_out
        # part_imgs的全连接分类层结果part_logits：batch_size × topN × 200
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)

        # 返回三个logits，前n个anchor的index和得分
        return [raw_logits, concat_logits, part_logits, top_n_index, top_n_prob]


def list_loss(logits, targets):
    """
    得到所有anchor为目标类别的分类概率值
    :param logits: (batch_size × PROPOSAL_NUM, -1)。-1对应的正是200
    :param targets: (batch_size × PROPOSAL_NUM, 1)。每张图片的label重复PROPOSAL_NUM次
    :return:loss:(batch_size * PROPOSAL_NUM, 1)。1为目标类别的分类概率值
    """
    temp = F.log_softmax(logits, -1)
    loss = [-temp[i][targets[i].item()] for i in range(logits.size(0))]  # loss:(batch_size * PROPOSAL_NUM)，每个元素为目标类别的分类概率值
    return torch.stack(loss)  # loss:(batch_size * PROPOSAL_NUM, 1)


def ranking_loss(score, targets, proposal_num=PROPOSAL_NUM):
    """
    误差是：更高概率的anchor，与本anchor，得分的差值。
    所以为了缩小loss，有两种途径：
    一、提高概率，使没有其他anchor比本anchor的概率还高；
    二、提高得分，使得分差值减少。
    :param score: 所有anchor的得分，也是top_n_prob：(batch_size, PROPOSAL_NUM)
    :param targets: 所有anchor目标类别的分类概率值：(batch_size, PROPOSAL_NUM)
    :param proposal_num: 每张图片的anchor数量
    :return: 一张图片所有anchor的loss
    """
    loss = torch.zeros(1).cuda()
    batch_size = score.size(0)
    for i in range(proposal_num):
        targets_p = (targets > targets[:, i].unsqueeze(1)).type(torch.cuda.FloatTensor)
        # targets_p：(batch_size, PROPOSAL_NUM)的0、1矩阵。1位置的anchor目标类别概率值更高
        pivot = score[:, i].unsqueeze(1)  # pivot:(batch_size, 1)
        loss_p = (1 - pivot + score) * targets_p
        # loss_p(batch_size, PROPOSAL_NUM)是score差值（score-pivot）的映射，将概率高于本anchor的差值保留下来，低于的清零
        loss_p = torch.sum(F.relu(loss_p))  # 把一个batch的loss加起来
        loss += loss_p
    return loss / batch_size
