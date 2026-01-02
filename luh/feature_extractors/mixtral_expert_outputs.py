import torch
import torch.nn.functional as F

from .feature_extractor_base import FeatureExtractorBase
from .utils import get_layer_nums


class FeatureExtractorMixtralExpertOutputs(FeatureExtractorBase):
    def __init__(self, orig_base_model, layer_nums="all", **kwargs):
        super().__init__()

        self._layer_nums = get_layer_nums(layer_nums, orig_base_model)

        try:
            decoder_layers = orig_base_model.model.layers
        except AttributeError as exc:  # pragma: no cover - defensive programming
            raise ValueError("Mixtral expert extractor requires a decoder-only model with `model.layers`.") from exc

        self._captures: dict[int, list[dict[str, torch.Tensor]]] = {
            layer_idx: [] for layer_idx in self._layer_nums
        }
        self._original_forwards: dict[int, callable] = {}

        moe_block = decoder_layers[0].block_sparse_moe
        self._hidden_size = orig_base_model.config.hidden_size
        self._num_experts = moe_block.num_experts
        self._top_k = moe_block.top_k

        # Feature dimension per layer: routing weights + expert ids (one-hot) + expert outputs
        per_layer_dim = self._top_k + self._top_k * self._num_experts + self._top_k * self._hidden_size
        self._feature_dim = len(self._layer_nums) * per_layer_dim

        self._patch_moe_blocks(decoder_layers)

    def _patch_moe_blocks(self, decoder_layers):
        for layer_idx, layer in enumerate(decoder_layers):
            if layer_idx not in self._layer_nums:
                continue

            block = layer.block_sparse_moe
            if getattr(block, "_luh_mixtral_wrapped", False):
                continue

            self._original_forwards[layer_idx] = block.forward

            def forward_with_capture(hidden_states, *, _block=block, _layer_idx=layer_idx):
                batch_size, sequence_length, hidden_dim = hidden_states.shape

                if _block.training and _block.jitter_noise > 0:
                    hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                        1.0 - _block.jitter_noise, 1.0 + _block.jitter_noise
                    )

                flat_states = hidden_states.view(-1, hidden_dim)
                router_logits = _block.gate(flat_states)

                routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
                routing_weights, selected_experts = torch.topk(routing_weights, _block.top_k, dim=-1)
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
                routing_weights = routing_weights.to(hidden_states.dtype)

                expert_mask = torch.nn.functional.one_hot(
                    selected_experts, num_classes=_block.num_experts
                ).permute(2, 1, 0)

                contributions = torch.zeros(
                    routing_weights.shape[0], _block.top_k, hidden_dim,
                    device=hidden_states.device, dtype=hidden_states.dtype,
                )

                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                for expert_idx in expert_hit:
                    expert_idx_int = expert_idx.item()
                    expert_layer = _block.experts[expert_idx_int]
                    idx, top_x = torch.where(expert_mask[expert_idx_int].squeeze(0))
                    current_state = flat_states[None, top_x].reshape(-1, hidden_dim)
                    current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                    contributions[top_x, idx] = current_hidden_states.to(hidden_states.dtype)

                final_hidden_states = contributions.sum(dim=1).reshape(batch_size, sequence_length, hidden_dim)

                # Store capture in a shape that is convenient for downstream feature construction
                capture = {
                    "routing_weights": routing_weights.view(batch_size, sequence_length, -1),
                    "selected_experts": selected_experts.view(batch_size, sequence_length, -1),
                    "expert_outputs": contributions.view(batch_size, sequence_length, _block.top_k, hidden_dim),
                }
                self._captures[_layer_idx].append(capture)

                # Keep the returned values consistent with the original forward signature
                return final_hidden_states, router_logits

            block.forward = forward_with_capture
            block._luh_mixtral_wrapped = True

    def _consume_captures(self):
        captures = self._captures
        self._captures = {layer_idx: [] for layer_idx in self._layer_nums}
        return captures

    def __call__(self, llm_inputs, llm_outputs):
        captures = self._consume_captures()
        if not captures or any(len(captures[layer]) == 0 for layer in self._layer_nums):
            raise ValueError("No Mixtral expert captures available. Ensure the base model was run before feature extraction.")

        is_training = not hasattr(llm_outputs, "sequences")

        layer_features = []
        for layer_idx in self._layer_nums:
            layer_captures = captures[layer_idx]

            routing_weights = torch.cat([c["routing_weights"] for c in layer_captures], dim=1)
            selected_experts = torch.cat([c["selected_experts"] for c in layer_captures], dim=1)
            expert_outputs = torch.cat([c["expert_outputs"] for c in layer_captures], dim=1)

            if is_training:
                routing_weights = routing_weights[:, :-1]
                selected_experts = selected_experts[:, :-1]
                expert_outputs = expert_outputs[:, :-1]
            else:
                routing_weights = routing_weights[:, 1:]
                selected_experts = selected_experts[:, 1:]
                expert_outputs = expert_outputs[:, 1:]

            expert_one_hot = torch.nn.functional.one_hot(
                selected_experts, num_classes=self._num_experts
            ).to(expert_outputs.dtype)

            layer_features.append(
                torch.cat(
                    [
                        routing_weights,
                        expert_one_hot.view(expert_one_hot.shape[0], expert_one_hot.shape[1], -1),
                        expert_outputs.view(expert_outputs.shape[0], expert_outputs.shape[1], -1),
                    ],
                    dim=-1,
                )
            )

        return torch.cat(layer_features, dim=-1)

    def feature_dim(self):
        return self._feature_dim


def load_extractor(config, base_model):
    return FeatureExtractorMixtralExpertOutputs(base_model, **config)
