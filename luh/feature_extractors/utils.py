import torch
from collections.abc import Iterable


def get_layer_nums(layer_nums, orig_base_model):
    if layer_nums == 'all':
        return list(range(orig_base_model.config.num_hidden_layers))
    elif isinstance(layer_nums, Iterable):
        return list(layer_nums)
    return (layer_nums,)


def get_head_nums(head_nums, layer_nums, orig_base_model):
    if head_nums == 'all':
        all_heads = list(range(orig_base_model.config.num_attention_heads))
        return {l: all_heads for l in layer_nums}
    elif isinstance(head_nums, dict):
        heads: dict[int, list[int]] = {}  # list of heads for each layer
        for key, val in head_nums.items():
            for l in get_layer_nums(key, orig_base_model):
                heads.update(get_head_nums(val, [l], orig_base_model))
        assert all(l in heads.keys() for l in layer_nums)
        return heads
    elif isinstance(head_nums, Iterable):
        return {l: list(head_nums) for l in layer_nums}
    return {l: (head_nums,) for l in layer_nums}


def get_hidden_states(llm_outputs, layer_nums=None, detach=True):
    """Return stacked hidden states only for the requested layers.

    Selecting a subset of layers avoids allocating a full
    ``batch x seq_len x num_layers x hidden`` tensor when only a few
    layers are needed. Detaching prevents autograd from retaining large
    computation graphs when backbone gradients are disabled.
    """

    hs = llm_outputs["hidden_states"]
    is_training = type(hs[-1]) == torch.Tensor

    if layer_nums is None:
        layer_nums = range(len(hs) if is_training else len(hs[0]))

    if is_training:
        stacked_layers = []
        for layer in layer_nums:
            layer_state = hs[layer]
            if detach:
                layer_state = layer_state.detach()
            stacked_layers.append(layer_state[:, :-1, :])
        return torch.stack(stacked_layers, dim=-2)

    per_token_layers = []
    for token_states in hs:
        stacked_layers = []
        for layer in layer_nums:
            layer_state = token_states[layer]
            if detach:
                layer_state = layer_state.detach()
            stacked_layers.append(layer_state)
        per_token_layers.append(torch.stack(stacked_layers, dim=-2))
    return torch.cat(per_token_layers, dim=1)
