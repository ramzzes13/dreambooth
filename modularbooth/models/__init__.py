"""Model components for ModularBooth multi-subject personalization.

Exports
-------
BlockwiseLoRA
    Adaptive-rank LoRA injection for DiT transformer blocks.
LoRALinear
    Low-rank adapter wrapping a single ``nn.Linear`` layer.
KnowledgeProbe
    Per-block probing to classify DiT blocks as identity / context / shared.
TokenAwareAttentionMask
    Spatial attention masking for multi-subject composition.
LoRAComposer
    Multi-subject LoRA loading and spatially-masked composition.
"""

from modularbooth.models.attention_mask import TokenAwareAttentionMask
from modularbooth.models.blockwise_lora import BlockwiseLoRA, LoRALinear
from modularbooth.models.knowledge_probe import KnowledgeProbe
from modularbooth.models.lora_merge import LoRAComposer

__all__ = [
    "BlockwiseLoRA",
    "LoRALinear",
    "KnowledgeProbe",
    "TokenAwareAttentionMask",
    "LoRAComposer",
]
