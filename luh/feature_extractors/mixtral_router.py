import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_extractor_base import FeatureExtractorBase
from .utils import get_layer_nums


class FeatureExtractorMixtralRouter(FeatureExtractorBase):
    """
    Extracts router logits/probabilities from Mixtral MoE layers and aggregates them
    into a fixed-size feature vector. The aggregated vector is then broadcast across
    the token dimension so it can be consumed by the uncertainty head, which expects
    a tensor shaped like ``[batch, seq_len, head_dim]``.
    """

    def __init__(
        self,
        orig_base_model,
        feature_source: str = "mixtral_router",
        router_output: str = "logits",
        token_agg: str = "mean",
        layer_agg: str = "mean",
        use_moe_layers=None,
        max_seq_len_for_concat_tokens: int | None = None,
        dim_align: str = "direct",
        target_head_dim: int | None = None,
        **kwargs,
    ):
        assert feature_source in {"mixtral_router", "dense_hidden"}, (
            "feature_source must be either 'mixtral_router' or 'dense_hidden' for backwards compatibility"
        )
        assert router_output in {"logits", "probs"}, "router_output must be 'logits' or 'probs'"
        assert token_agg in {"mean", "concat"}, "token_agg must be 'mean' or 'concat'"
        assert layer_agg in {"mean", "concat"}, "layer_agg must be 'mean' or 'concat'"
        assert dim_align in {"direct", "projection"}, "dim_align must be 'direct' or 'projection'"

        # Dense hidden state extraction is still supported by falling back to the last hidden state.
        self._feature_source = feature_source
        if self._feature_source == "dense_hidden":
            self._projection = None
            self._output_dim = orig_base_model.config.hidden_size
            return

        self._router_output = router_output
        self._token_agg = token_agg
        self._layer_agg = layer_agg
        self._target_head_dim = target_head_dim
        self._dim_align = dim_align

        # Determine which MoE layers to read. Defaults to all layers.
        self._layer_indices = get_layer_nums(
            use_moe_layers if use_moe_layers is not None else "all", orig_base_model
        )

        # Number of experts is read from the Mixtral config.
        self._num_experts = getattr(orig_base_model.config, "num_experts", None)
        if self._num_experts is None:
            raise ValueError("Mixtral router feature extractor requires num_experts in the model config")

        # Token aggregation determines per-layer feature width.
        if self._token_agg == "mean":
            per_layer_dim = self._num_experts
        else:
            if max_seq_len_for_concat_tokens is None:
                raise ValueError(
                    "max_seq_len_for_concat_tokens must be provided when token_agg='concat'"
                )
            self._max_seq_len_for_concat_tokens = max_seq_len_for_concat_tokens
            per_layer_dim = self._num_experts * self._max_seq_len_for_concat_tokens

        # Layer aggregation determines final raw feature dim before alignment.
        # - mean_over_layers: [batch, per_layer_dim]
        # - concat_over_layers: [batch, num_layers * per_layer_dim]
        if self._layer_agg == "mean":
            self._raw_feature_dim = per_layer_dim
        else:
            self._raw_feature_dim = per_layer_dim * len(self._layer_indices)

        # Alignment with the uncertainty head expectation.
        if self._dim_align == "direct":
            if self._target_head_dim is None:
                raise ValueError(
                    "target_head_dim must be provided when dim_align='direct' to validate feature dimensions"
                )
            if self._raw_feature_dim != self._target_head_dim:
                raise ValueError(
                    f"Router feature dim ({self._raw_feature_dim}) does not match head_dim ({self._target_head_dim})"
                )
            self._output_dim = self._raw_feature_dim
            self._projection = None
        else:
            if self._target_head_dim is None:
                raise ValueError(
                    "target_head_dim must be provided when dim_align='projection' to build the alignment layer"
                )
            self._projection = nn.Linear(self._raw_feature_dim, self._target_head_dim)
            self._output_dim = self._target_head_dim

    def _get_token_mask(self, llm_inputs, llm_outputs, seq_len: int):
        if hasattr(llm_outputs, "full_attention_mask"):
            mask = llm_outputs.full_attention_mask
        else:
            mask = llm_inputs["attention_mask"]
        return mask[:, :seq_len]

    def _get_head_attention_mask(self, llm_inputs, llm_outputs):
        if hasattr(llm_outputs, "sequences"):
            return llm_outputs["full_attention_mask"][:, 1:]
        return llm_inputs["attention_mask"][:, :-1]

    def _collect_router_outputs(self, llm_outputs):
        router_key = "router_probs" if self._router_output == "probs" else "router_logits"
        router_values = getattr(llm_outputs, router_key, None)
        if router_values is None and isinstance(llm_outputs, dict):
            router_values = llm_outputs.get(router_key)
        logits_fallback = False
        if router_values is None and self._router_output == "probs":
            router_values = getattr(llm_outputs, "router_logits", None)
            if router_values is None and isinstance(llm_outputs, dict):
                router_values = llm_outputs.get("router_logits")
            logits_fallback = router_values is not None
        if router_values is None:
            raise ValueError(
                f"Mixtral outputs must include {router_key}. Ensure output_router_logits=True in the base model call."
            )

        if isinstance(router_values, torch.Tensor):
            per_layer = [router_values]
        elif isinstance(router_values, (list, tuple)):
            per_layer = list(router_values)
        else:
            raise TypeError(f"Unexpected router output type: {type(router_values)}")

        selected_layers = [per_layer[i] for i in self._layer_indices]

        if self._router_output == "probs" and (logits_fallback or not hasattr(llm_outputs, "router_probs")):
            # Router probabilities are derived from logits when not returned explicitly.
            selected_layers = [F.softmax(layer, dim=-1) for layer in selected_layers]

        return torch.stack(selected_layers, dim=1)  # [batch, n_layers, seq_len, num_experts]

    def _aggregate_tokens(self, router_tensor, mask):
        """
        router_tensor: [batch, n_layers, seq_len, num_experts]
        mask: [batch, seq_len]
        """
        if self._token_agg == "mean":
            masked = router_tensor * mask.unsqueeze(1).unsqueeze(-1)
            token_counts = mask.sum(dim=-1).clamp(min=1).unsqueeze(1).unsqueeze(-1)
            # Resulting shape: [batch, n_layers, num_experts]
            return masked.sum(dim=2) / token_counts

        # token concat: pad/truncate to fixed max_seq_len_for_concat_tokens
        seq_len = router_tensor.shape[2]
        max_len = self._max_seq_len_for_concat_tokens
        if seq_len < max_len:
            pad_len = max_len - seq_len
            router_tensor = torch.nn.functional.pad(router_tensor, (0, 0, 0, pad_len))
            mask = torch.nn.functional.pad(mask, (0, pad_len))
        elif seq_len > max_len:
            router_tensor = router_tensor[:, :, :max_len, :]
            mask = mask[:, :max_len]

        masked = router_tensor * mask.unsqueeze(1).unsqueeze(-1)
        # Shape becomes [batch, n_layers, max_len * num_experts]
        return masked.flatten(start_dim=2)

    def _aggregate_layers(self, token_features):
        """
        token_features shapes:
        - token_agg == mean: [batch, n_layers, num_experts]
        - token_agg == concat: [batch, n_layers, max_seq_len * num_experts]
        """
        if self._layer_agg == "mean":
            return token_features.mean(dim=1)
        return token_features.flatten(start_dim=1)

    def __call__(self, llm_inputs, llm_outputs):
        if self._feature_source == "dense_hidden":
            # Backwards-compatible path that returns the last hidden state features.
            hidden_states = llm_outputs["hidden_states"][-1]
            return hidden_states[:, :-1, :]

        router_tensor = self._collect_router_outputs(llm_outputs)
        # router_tensor: [batch, n_layers, seq_len, num_experts]
        token_mask = self._get_token_mask(llm_inputs, llm_outputs, router_tensor.shape[2])
        token_features = self._aggregate_tokens(router_tensor, token_mask)
        # token_features: per-layer token aggregation
        layer_features = self._aggregate_layers(token_features)
        # layer_features: [batch, raw_feature_dim]

        if self._projection is not None:
            layer_features = self._projection(layer_features)

        # Broadcast to match per-token expectations of the uncertainty heads.
        head_attn_mask = self._get_head_attention_mask(llm_inputs, llm_outputs)
        seq_len = head_attn_mask.shape[1]
        expanded = layer_features.unsqueeze(1).expand(-1, seq_len, -1)
        return expanded

    def feature_dim(self):
        return self._output_dim

    def requires_router_outputs(self):
        return self._feature_source == "mixtral_router"


def load_extractor(config, base_model, **_kwargs):
    return FeatureExtractorMixtralRouter(base_model, **config)
