import pytest
import torch

from text_lm.generate import generate_token_ids


class IncrementingModel:
    max_seq_len = 3

    def __init__(self):
        self.max_seen_sequence_length = 0

    def __call__(self, input_ids):
        self.max_seen_sequence_length = max(
            self.max_seen_sequence_length, input_ids.shape[1],
        )
        vocab_size = 5
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], vocab_size),
            float("-inf"),
        )
        next_ids = (input_ids[:, -1] + 1) % vocab_size
        logits[:, -1, :].scatter_(1, next_ids[:, None], 0.0)
        return logits


def test_generate_token_ids_uses_sliding_context_and_greedy_decoding():
    model = IncrementingModel()

    generated = generate_token_ids(
        model,
        torch.tensor([[1, 2]]),
        max_new_tokens=5,
    )

    assert generated.tolist() == [[1, 2, 3, 4, 0, 1, 2]]
    assert model.max_seen_sequence_length == 3


def test_generate_token_ids_stops_at_eos():
    model = IncrementingModel()

    generated = generate_token_ids(
        model,
        torch.tensor([[3]]),
        max_new_tokens=5,
        eos_token_id=4,
    )

    assert generated.tolist() == [[3, 4]]


@pytest.mark.parametrize("max_new_tokens", [0, 1025])
def test_generate_token_ids_rejects_generation_over_1024_tokens(max_new_tokens):
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate_token_ids(
            IncrementingModel(),
            torch.tensor([[1]]),
            max_new_tokens=max_new_tokens,
        )
