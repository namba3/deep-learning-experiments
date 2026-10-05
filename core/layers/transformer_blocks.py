import torch.nn as nn
from .attention import CausalMHLA, GatedMultiheadAttention, GroupedQueryAttention, InterpolatedGatedMultiheadAttention, KVSelfAttention
from .feedforward import GatedLinear, InterpolatedGatedLinear, InterpolatedSuperLinear



class TransformerBlock(nn.Module):
    """
    Transformerの1ブロック
    - pre-norm構成
    - 正規化層をLayerNormをRMSNormに変更
    - Multihead AttentionをGated Multihead Attentionに変更
    - FFNにGatedLinearを採用
    - FFNの中間dimには入力の4倍が使われることが多いが、GatedLinearを使う場合は小さなdimでも良い(元の2/3倍程度が推奨されている)
    - 残差接続の加算部分にDropoutを追加
    - BatchFirst前提
    - 入力xの形状: (B, N, D)
    - 出力の形状: (B, N, D)
    """
    def __init__(self, embed_dim, num_heads, drop_out=0.1, rope_base=None,
                 residual_scale=1.0):
        super().__init__()
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
        self.residual_scale = residual_scale
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = GatedMultiheadAttention(
            embed_dim,
            num_heads,
            dropout=drop_out,
            batch_first=True,
            rope_base=rope_base,
        )
        self.attn_dropout = nn.Dropout(drop_out)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.ffn = nn.Sequential(
            GatedLinear(embed_dim, embed_dim * 3),
            nn.Dropout(drop_out),
            nn.Linear(embed_dim * 3, embed_dim),
        )
        self.ffn_dropout = nn.Dropout(drop_out)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, N, D)
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x2 = x + self.residual_scale * self.attn_dropout(attn_output)
        x2_norm = self.norm2(x2)
        ffn_output = self.ffn(x2_norm)
        out = x2 + self.residual_scale * self.ffn_dropout(ffn_output)
        return out

class GatedGQATransformerBlock(nn.Module):
    """Pre-norm causal Transformer block with query-gated GQA."""
    def __init__(self, embed_dim, num_heads, kv_heads, drop_out=0.1,
                 rope_base=10000.0, residual_scale=1.0):
        super().__init__()
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
        self.residual_scale = residual_scale
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = GroupedQueryAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kv_heads=kv_heads,
            dropout=drop_out,
            batch_first=True,
            gate_mode="query",
            rope_base=rope_base,
        )
        self.attn_dropout = nn.Dropout(drop_out)
        self.norm2 = nn.RMSNorm(embed_dim)
        hidden_dim = embed_dim * 3
        self.ffn = nn.Sequential(
            GatedLinear(embed_dim, hidden_dim),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.ffn_dropout = nn.Dropout(drop_out)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x_norm = self.norm1(x)
        attn_output = self.attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + self.residual_scale * self.attn_dropout(attn_output)
        x_norm = self.norm2(x)
        return x + self.residual_scale * self.ffn_dropout(self.ffn(x_norm))

class MHLATransformerBlock(nn.Module):
    """Pre-norm causal Transformer block using sequence MHLA."""
    def __init__(self, embed_dim, num_heads, kv_heads, drop_out=0.1,
                 rope_base=10000.0, residual_scale=1.0):
        super().__init__()
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")
        self.residual_scale = residual_scale
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = CausalMHLA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kv_heads=kv_heads,
            dropout=drop_out,
            rope_base=rope_base,
        )
        self.attn_dropout = nn.Dropout(drop_out)
        self.norm2 = nn.RMSNorm(embed_dim)
        hidden_dim = embed_dim * 3
        self.ffn = nn.Sequential(
            GatedLinear(embed_dim, hidden_dim),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.ffn_dropout = nn.Dropout(drop_out)

    def forward(self, x, key_padding_mask=None):
        x_norm = self.norm1(x)
        attn_output = self.attn(
            x_norm, key_padding_mask=key_padding_mask,
        )
        x = x + self.residual_scale * self.attn_dropout(attn_output)
        x_norm = self.norm2(x)
        return x + self.residual_scale * self.ffn_dropout(self.ffn(x_norm))

class InterpolatedTransformerBlock(nn.Module):
    """Q/K/V/FFN の重みを depth_step から生成する Transformer Block。"""
    def __init__(self, embed_dim, num_heads, num_steps=4, drop_out=0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = InterpolatedGatedMultiheadAttention(
            embed_dim, num_heads, num_steps=num_steps,
            dropout=drop_out, batch_first=True
        )
        self.attn_dropout = nn.Dropout(drop_out)
        self.norm2 = nn.RMSNorm(embed_dim)
        hidden_dim = embed_dim * 3
        self.ffn_in = InterpolatedGatedLinear(
            embed_dim, hidden_dim, num_steps=num_steps, bias=False
        )
        self.ffn_dropout_layer = nn.Dropout(drop_out)
        self.ffn_out = InterpolatedSuperLinear(
            hidden_dim, embed_dim, num_steps=num_steps, bias=False
        )
        self.ffn_dropout = nn.Dropout(drop_out)

    def forward(self, x, depth_step):
        x_norm = self.norm1(x)
        attn_out = self.attn(
            x_norm, x_norm, x_norm, depth_step=depth_step
        )
        x = x + self.attn_dropout(attn_out)

        x_norm = self.norm2(x)
        ffn_out = self.ffn_in(x_norm, depth_step)
        ffn_out = self.ffn_dropout_layer(ffn_out)
        ffn_out = self.ffn_out(ffn_out, depth_step)
        return x + self.ffn_dropout(ffn_out)

class KVTransformerBlock(nn.Module):
    """
    Qを廃止したKV Transformer Block
    - pre-norm構成
    - 正規化層はRMSNorm
    - AttentionはQなし・K/Vのみ（GatedLinear）
    - FFNにもGatedLinearを採用
    - 内部の Linear 層では bias 項無し; 残差接続とRMSNormがbiasを吸収する
    - 残差接続にDropoutを挿入
    - BatchFirst前提
    - 入力/出力形状: (B, N, D)
    """
    def __init__(self, embed_dim, num_heads, drop_out=0.1):
        super().__init__()

        self.norm1 = nn.RMSNorm(embed_dim)

        self.attn = KVSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=drop_out,
            batch_first=True,
        )

        self.attn_dropout = nn.Dropout(drop_out)

        self.norm2 = nn.RMSNorm(embed_dim)

        # Gated FFN
        hidden_dim = int(embed_dim * 3)  # GatedLinear前提で小さめ
        self.ffn = nn.Sequential(
            GatedLinear(embed_dim, hidden_dim, bias=False),
            nn.Dropout(drop_out),
            nn.Linear(hidden_dim, embed_dim, bias=False),
        )

        self.ffn_dropout = nn.Dropout(drop_out)

    def forward(self, x):
        # x: (B, N, D)

        # --- KV Self-Attention ---
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm)
        x = x + self.attn_dropout(attn_out)

        # --- FFN ---
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.ffn_dropout(ffn_out)

        return x
