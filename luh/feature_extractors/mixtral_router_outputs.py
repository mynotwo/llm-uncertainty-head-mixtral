import torch
import torch.nn as nn

from .feature_extractor_base import FeatureExtractorBase
from .utils import get_layer_nums


class FeatureExtractorMixtralRouterOutputs(FeatureExtractorBase):
    def __init__(
        self,
        orig_base_model,
        concat_mode: str = "feature",
        projection_dim: int | None = None,
        layer_nums="all",
        **kwargs,
    ):
        self._layer_nums = get_layer_nums(layer_nums, orig_base_model)
        self._concat_mode = concat_mode

        num_experts = getattr(
            orig_base_model.config,
            "num_local_experts",
            getattr(orig_base_model.config, "num_experts", None),
        )
        if num_experts is None:
            raise ValueError("Mixtral router logits are unavailable: num_experts is undefined in model config")

        raw_dim = num_experts * len(self._layer_nums) if concat_mode == "feature" else num_experts

        self._projection = None
        if projection_dim is not None and projection_dim != raw_dim:
            self._projection = nn.Linear(raw_dim, projection_dim)
            self._feature_dim = projection_dim
        else:
            self._feature_dim = raw_dim

    def _get_router_logits(self, llm_outputs):
        if hasattr(llm_outputs, "router_logits"):
            router_logits = getattr(llm_outputs, "router_logits")
        elif isinstance(llm_outputs, dict) and "router_logits" in llm_outputs:
            router_logits = llm_outputs["router_logits"]
        else:
            raise ValueError("router_logits are missing from model outputs. Make sure output_router_logits=True is set")

        if not isinstance(router_logits, (list, tuple)):
            raise ValueError("router_logits are expected to be a list or tuple over layers")
        return router_logits

    def _prepare_router_logits(self, llm_outputs, is_training: bool):
        router_logits = self._get_router_logits(llm_outputs)

        if not is_training and isinstance(router_logits[0], (list, tuple)):
            num_layers = len(router_logits[0])
            stacked_layers = [[] for _ in range(num_layers)]
            for step_router in router_logits:
                for i, layer_router in enumerate(step_router):
                    stacked_layers[i].append(layer_router)
            return [torch.cat(layer_steps, dim=1) for layer_steps in stacked_layers]

        processed = list(router_logits)
        if is_training:
            processed = [layer[:, :-1, :] for layer in processed]
        return processed

    def _base_attention_mask(self, llm_inputs, llm_outputs, is_training: bool):
        if is_training:
            return llm_inputs["attention_mask"][:, :-1]
        if isinstance(llm_outputs, dict):
            return llm_outputs["full_attention_mask"][:, 1:]
        return getattr(llm_outputs, "full_attention_mask")[:, 1:]

    def _set_feature_mask(self, llm_outputs, mask):
        if isinstance(llm_outputs, dict):
            llm_outputs["feature_attention_mask"] = mask
        else:
            setattr(llm_outputs, "feature_attention_mask", mask)

    def __call__(self, llm_inputs, llm_outputs):
        is_training = not hasattr(llm_outputs, "sequences")
        router_logits = self._prepare_router_logits(llm_outputs, is_training)
        selected_layers = [router_logits[i] for i in self._layer_nums]
        base_mask = self._base_attention_mask(llm_inputs, llm_outputs, is_training)

        if self._concat_mode == "sequence":
            stacked = torch.stack(selected_layers, dim=2)  # (batch, seq_len, num_layers, num_experts)
            features = stacked.reshape(stacked.size(0), -1, stacked.size(-1))

            expanded_mask = base_mask.unsqueeze(-1).expand(-1, -1, len(self._layer_nums))
            expanded_mask = expanded_mask.reshape(expanded_mask.size(0), -1)
            self._set_feature_mask(llm_outputs, expanded_mask)
        else:
            features = torch.cat(selected_layers, dim=-1)
            if features.size(1) != base_mask.size(1):
                raise ValueError(
                    f"Router logits length {features.size(1)} does not match attention mask length {base_mask.size(1)}"
                )

        if self._projection is not None:
            features = self._projection(features)

        return features

    def feature_dim(self):
        return self._feature_dim


def load_extractor(config, base_model):
    return FeatureExtractorMixtralRouterOutputs(base_model, **config)
