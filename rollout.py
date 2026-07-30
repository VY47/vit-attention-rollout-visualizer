import torch

def attention_rollout(attentions, discard_ratio=0.9, head_fusion="mean"):
    """
    attentions: list of tensors, one per layer, shape [1, heads, tokens, tokens]
    Returns: rollout attention map for the CLS token, shape [num_patches]
    """
    result = torch.eye(attentions[0].size(-1))  # start with identity

    with torch.no_grad():
        for attention in attentions:
            attention = attention[0]  # remove batch dim -> [heads, tokens, tokens]

            # fuse attention heads
            if head_fusion == "mean":
                fused = attention.mean(dim=0)
            elif head_fusion == "max":
                fused = attention.max(dim=0)[0]
            else:
                fused = attention.min(dim=0)[0]

            # discard the lowest attentions (keep only strongest connections)
            flat = fused.view(-1)
            _, indices = flat.topk(int(flat.size(0) * (1 - discard_ratio)), largest=True)
            mask = torch.zeros_like(flat)
            mask[indices] = flat[indices]
            fused = mask.view(fused.size())

            # account for residual connections (add identity, then renormalize)
            I = torch.eye(fused.size(-1))
            fused = (fused + I) / 2
            fused = fused / fused.sum(dim=-1, keepdim=True)

            result = torch.matmul(fused, result)

    # CLS token's attention to all patches (excluding itself)
    mask = result[0, 1:]
    return mask