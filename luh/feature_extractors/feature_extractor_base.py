from abc import ABC, abstractmethod


class FeatureExtractorBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, llm_inputs, llm_outputs):
        pass

    @abstractmethod
    def feature_dim(self):
        pass

    def output_attention(self):
        return False

    def requires_hidden_states(self):
        """Whether this extractor needs ``output_hidden_states`` from the backbone."""
        return False
