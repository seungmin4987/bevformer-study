import torch, math
import torch.nn as nn

class SinePositionalEncoding(nn.Module):
    def __init__(self, num_feats, temperature=10000, normalize=False,
                 scale=2*math.pi, eps=1e-6, offset=0.):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):  # mask: [B, H, W]
        not_mask = 1 - mask.int()
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).view(mask.size(0), mask.size(1), mask.size(2), -1)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).view(mask.size(0), mask.size(1), mask.size(2), -1)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class LearnedPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights (torch-only).
    
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension is 2 * num_feats.
        row_num_embed (int): Dictionary size of row embeddings (height).
        col_num_embed (int): Dictionary size of col embeddings (width).
    """
    def __init__(self, num_feats, row_num_embed=50, col_num_embed=50):
        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        
        # 파라미터 초기화 (기본 uniform)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask (Tensor): shape [B, H, W] (values ignored, only size matters).
        Returns:
            pos (Tensor): shape [B, 2*num_feats, H, W]
        """
        B, H, W = mask.shape
        device = mask.device

        x = torch.arange(W, device=device)  # [W]
        y = torch.arange(H, device=device)  # [H]

        x_embed = self.col_embed(x)  # [W, num_feats]
        y_embed = self.row_embed(y)  # [H, num_feats]

        # (H, W, 2*num_feats)
        pos = torch.cat([
            x_embed.unsqueeze(0).repeat(H, 1, 1),   # [H, W, num_feats]
            y_embed.unsqueeze(1).repeat(1, W, 1)    # [H, W, num_feats]
        ], dim=-1)

        # [B, 2*num_feats, H, W]
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)
        return pos

    def __repr__(self):
        return (f"{self.__class__.__name__}(num_feats={self.num_feats}, "
                f"row_num_embed={self.row_num_embed}, "
                f"col_num_embed={self.col_num_embed})")

