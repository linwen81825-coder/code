import torch
import torch.nn as nn

from .resnet import ResNetBackbone


class ExpertFFN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TopKGating(nn.Module):
    def __init__(self, in_dim, num_experts, topk):
        super().__init__()
        self.topk = topk
        self.gate = nn.Linear(in_dim, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        probs = torch.softmax(logits.float(), dim=-1)
        topk_vals, topk_idx = probs.topk(self.topk, dim=-1)
        weights = torch.zeros_like(probs)
        weights.scatter_(1, topk_idx, topk_vals)
        weights = weights.to(x.dtype)
        return weights, topk_idx


class MoELayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_experts, topk):
        super().__init__()
        self.gating = TopKGating(in_dim, num_experts, topk)
        self.experts = nn.ModuleList(
            [ExpertFFN(in_dim, hidden_dim, out_dim) for _ in range(num_experts)]
        )

    def forward(self, x, return_stats=False):
        weights, topk_idx = self.gating(x)
        B = x.size(0)
        C = self.experts[0].fc2.out_features
        out = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        sample_hits_by_expert = None
        if return_stats:
            sample_hits_by_expert = torch.zeros(
                B,
                len(self.experts),
                device=x.device,
                dtype=torch.long,
            )

        for i, expert in enumerate(self.experts):
            expert_mask = topk_idx == i
            if sample_hits_by_expert is not None:
                sample_hits_by_expert[:, i] = expert_mask.sum(dim=-1)
            token_mask = expert_mask.any(dim=-1)
            if not token_mask.any():
                continue
            expert_out = expert(x[token_mask])
            sel_weights = weights[token_mask, i]
            out[token_mask] += expert_out * sel_weights.unsqueeze(-1)

        if return_stats:
            stats = {
                "expert_activations": sample_hits_by_expert.sum(dim=0),
                "sample_hits_by_expert": sample_hits_by_expert,
                "avg_router_probs": weights.detach().mean(dim=0),
                "capacity": B,
            }
            return out, stats

        return out


class MoEFedModel(nn.Module):
    def __init__(self, in_channels, num_classes, img_size, num_experts, topk):
        super().__init__()
        self.backbone = ResNetBackbone(in_channels, img_size)
        feat_dim = self.backbone.feat_dim
        self.moe_head = MoELayer(feat_dim, 512, num_classes, num_experts, topk)

    def forward(self, x, return_stats=False):
        feat = self.backbone(x)
        if return_stats:
            logits, stats = self.moe_head(feat, return_stats=True)
            return {
                "logits": logits,
                "expert_stats_by_layer": {"moe_head": stats},
            }
        logits = self.moe_head(feat)
        return logits
