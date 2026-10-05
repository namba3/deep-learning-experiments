import torch
import torch.nn.functional as F

from image_gen.train import GroupedQueryProjection


def test_grouped_query_projection_loads_legacy_q_kv_state():
    torch.manual_seed(0)
    module = GroupedQueryProjection(dim=16, heads=4, kv_heads=2).float()
    q_weight = torch.randn_like(module.qkv.weight[:16])
    q_bias = torch.randn_like(module.qkv.bias[:16])
    kv_weight = torch.randn_like(module.qkv.weight[16:])
    kv_bias = torch.randn_like(module.qkv.bias[16:])
    legacy_state = {
        "q.weight": q_weight,
        "q.bias": q_bias,
        "kv.weight": kv_weight,
        "kv.bias": kv_bias,
    }

    module.load_state_dict(legacy_state, strict=True)
    x = torch.randn(2, 5, 16)
    query, key, value = module(x)
    expected_query = F.linear(x, q_weight, q_bias).reshape(
        2, 5, 4, 4,
    ).transpose(1, 2)
    expected_key, expected_value = F.linear(x, kv_weight, kv_bias).chunk(
        2, dim=-1,
    )
    expected_key = expected_key.reshape(2, 5, 2, 4).transpose(1, 2)
    expected_value = expected_value.reshape(2, 5, 2, 4).transpose(1, 2)
    assert torch.allclose(query, expected_query)
    assert torch.allclose(key, expected_key)
    assert torch.allclose(value, expected_value)


def test_grouped_query_projection_uses_one_fused_parameter():
    module = GroupedQueryProjection(dim=16, heads=4, kv_heads=2)
    assert set(module.state_dict()) == {"qkv.weight", "qkv.bias"}
