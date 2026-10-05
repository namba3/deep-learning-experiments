import torch
import torch.nn as nn
from .attention import GatedMultiheadAttention, GroupedQueryAttention, KVSelfAttention



class AttentionPoolingWithGMHA(nn.Module):
    """
    Gated Multihead Attentionを用いた特徴量の集約
    - 学習可能なQueryベクトルを用いて、入力特徴量から重要な情報を抽出
    - BatchFirst前提
    - 入力Hの形状: (batch, seq_len, hidden_dim)
    - 出力: (batch, hidden_dim), (batch, 1, seq_len)（集約後の特徴量とAttention重み）
    """
    def __init__(self, hidden_dim, num_heads,dropout=0.1):
        super().__init__()
        self.query_vec = nn.Parameter(torch.randn(1, 1, hidden_dim))  # 学習可能な集約用Query
        self.attn = GatedMultiheadAttention(embed_dim=hidden_dim,
                                          num_heads=num_heads,
                                          batch_first=True,
                                          dropout=dropout)

    def forward(self, H):  # H: (batch, seq_len, hidden_dim)
        batch_size = H.size(0)
        # Queryをbatch分に複製
        Q = self.query_vec.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
        # MHAに通す（出力: pooled, attn_weights）
        pooled, attn_weights = self.attn(Q, H, H)  # pooled: (batch, 1, hidden_dim)
        return pooled.squeeze(1), attn_weights

class AttentionPoolingWithKVSelfAttention(nn.Module):
    """
    KVSelfAttention を用いた特徴量集約
    - 学習可能な pool token を系列に追加
    - Q は一切使用しない
    - BatchFirst 前提
    - 入力:  (B, N, D)
    - 出力:  (B, D), (B, 1, N)
    """
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()

        self.pool_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.attn = KVSelfAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=False,
            batch_first=True,
        )

    def forward(self, H):
        # H: (B, N, D)
        B, N, D = H.shape

        # pool token を batch 分複製
        pool = self.pool_token.expand(B, 1, D)

        # 系列の先頭に追加
        x = torch.cat([pool, H], dim=1)  # (B, N+1, D)

        # KV Self-Attention
        out, attn_weights = self.attn(x, need_weights=True)

        # pool token の出力のみ取得
        pooled = out[:, 0]  # (B, D)

        # pool token が各トークンをどれだけ見たか
        # attn_weights: (B, N+1, N+1)
        pool_attn = attn_weights[:, 0:1, 1:]  # (B, 1, N)

        return pooled, pool_attn

class AttentionPoolingWithGroupedQueryAttention(nn.Module):
    """Pool image tokens with a learned query and grouped K/V heads."""
    def __init__(self, hidden_dim, num_heads, kv_heads=None, dropout=0.1):
        super().__init__()
        self.pool_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = GroupedQueryAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            kv_heads=kv_heads,
            dropout=dropout,
            bias=False,
            batch_first=True,
        )

    def forward(self, hidden_states):
        batch = hidden_states.shape[0]
        pool = self.pool_token.expand(batch, -1, -1)
        pooled = self.attn(pool, hidden_states, hidden_states)
        # Native fused SDPA does not materialize attention maps.  Keep the
        # two-value pooling API for callers that previously ignored the map.
        return pooled[:, 0], None
