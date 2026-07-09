import torch
import torch.nn as nn
import torch.nn.functional as F


class CETripletLoss(nn.Module):
    def __init__(self, batch_size, ce_loss_cfg, triplet_cfg, focal_loss_cfg):
        super(CETripletLoss, self).__init__()
        self.batch_size = batch_size
        
        raw_weight = ce_loss_cfg.get("class_weights", None)
        
        if raw_weight is not None:
            self.class_weights = torch.tensor(raw_weight, dtype=torch.float).cuda()
        else:
            self.class_weights = None
        
        self.ce_enabled = ce_loss_cfg.get('enabled', True)
        self.ce_weight = ce_loss_cfg.get('weight', 1.0) if self.ce_enabled else 0.
        self.label_smoothing = ce_loss_cfg.get('label_smoothing', 0.)
        self.ce_reduction = ce_loss_cfg.get('reduction', "mean")
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction=self.ce_reduction, label_smoothing=self.label_smoothing)
        
        self.triplet_enabled = triplet_cfg.get('enabled', True)
        self.triplet_weight = triplet_cfg.get('weight', 1.0) if self.triplet_enabled else 0.
        self.P, self.M = batch_size
        self.hard_or_full_trip = triplet_cfg.get("hard_or_full", 'full')
        self.margin = triplet_cfg.get("margin", 0.2)
        self.triplet_loss = TripletLoss(self.P * self.M, self.hard_or_full_trip, self.margin)

        focal_raw_weight = focal_loss_cfg.get("class_weights", None)
        
        if focal_raw_weight is not None:
            self.focal_class_weights = torch.tensor(focal_raw_weight, dtype=torch.float).cuda()
        else:
            self.focal_class_weights = None

        self.focal_enabled = focal_loss_cfg.get('enabled', True)
        self.focal_weight = focal_loss_cfg.get('weight', 1.0) if self.focal_enabled else 0.
        self.focal_reduction = focal_loss_cfg.get('reduction', "mean")
        self.gamma = focal_loss_cfg.get("gamma", 2.)
        self.alpha = focal_loss_cfg.get("alpha", 0.25)
        self.focal_loss = FocalLoss(class_weights=self.focal_class_weights, gamma=self.gamma, alpha=self.alpha, reduction=self.focal_reduction)

    def forward(self, logits, features, targets, pid_targets):
        loss = 0.
        
        if self.ce_enabled:
            loss_ce = self.ce_loss(logits, targets)
            loss += self.ce_weight * loss_ce

        if self.triplet_enabled:
            # features: [N, n_parts, d] -> [n_parts, N, d]
            triplet_feature = features.permute(1, 0, 2).contiguous()
            triplet_label = pid_targets.unsqueeze(0).repeat(triplet_feature.size(0), 1)

            full_loss_metric_mean, hard_loss_metric_mean, mean_dist, full_loss_num = \
                self.triplet_loss(triplet_feature, triplet_label)

            if self.hard_or_full_trip == 'full':
                triplet_loss = full_loss_metric_mean.mean()
            elif self.hard_or_full_trip == 'hard':
                triplet_loss = hard_loss_metric_mean.mean()
            else:
                raise ValueError(f"hard_or_full = 'hard' | 'full' but hard_or_full = {self.hard_or_full_trip}")
            
            loss += self.triplet_weight * triplet_loss

        if self.focal_enabled:
            fl = self.focal_loss(logits, targets)
            loss += self.focal_weight * fl

        return loss
        

class TripletLoss(nn.Module):
    def __init__(self, batch_size, hard_or_full, margin):
        super(TripletLoss, self).__init__()
        self.batch_size = batch_size
        self.margin = margin

    def forward(self, feature, label):
        # feature: [n, m, d], label: [n, m]
        n, m, d = feature.size()
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).bool().view(-1)
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).bool().view(-1)

        dist = self.batch_dist(feature)
        mean_dist = dist.mean(1).mean(1)
        dist = dist.view(-1)
        # hard
        hard_hp_dist = torch.max(torch.masked_select(dist, hp_mask).view(n, m, -1), 2)[0]
        hard_hn_dist = torch.min(torch.masked_select(dist, hn_mask).view(n, m, -1), 2)[0]
        hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(n, -1)

        hard_loss_metric_mean = torch.mean(hard_loss_metric, 1)

        # non-zero full
        full_hp_dist = torch.masked_select(dist, hp_mask).view(n, m, -1, 1)
        full_hn_dist = torch.masked_select(dist, hn_mask).view(n, m, 1, -1)
        full_loss_metric = F.relu(self.margin + full_hp_dist - full_hn_dist).view(n, -1)

        full_loss_metric_sum = full_loss_metric.sum(1)
        full_loss_num = (full_loss_metric != 0).sum(1).float()

        full_loss_metric_mean = full_loss_metric_sum / full_loss_num
        full_loss_metric_mean[full_loss_num == 0] = 0

        return full_loss_metric_mean, hard_loss_metric_mean, mean_dist, full_loss_num

    def batch_dist(self, x):
        x2 = torch.sum(x ** 2, 2)
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        return dist

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean', class_weights=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.class_weights = class_weights
 
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
 
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss