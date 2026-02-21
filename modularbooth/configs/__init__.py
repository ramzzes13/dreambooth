"""Configuration management for ModularBooth."""

from pathlib import Path

from omegaconf import OmegaConf, DictConfig

CONFIGS_DIR = Path(__file__).parent


def load_config(backbone: str = "flux", overrides: dict | None = None) -> DictConfig:
    """Load and merge configuration files.

    Args:
        backbone: Which backbone config to use ("flux" or "sd3").
        overrides: Optional dict of overrides to apply on top.

    Returns:
        Merged OmegaConf DictConfig.
    """
    default_cfg = OmegaConf.load(CONFIGS_DIR / "default.yaml")
    backbone_cfg_path = CONFIGS_DIR / f"{backbone}.yaml"

    if backbone_cfg_path.exists():
        backbone_cfg = OmegaConf.load(backbone_cfg_path)
        cfg = OmegaConf.merge(default_cfg, backbone_cfg)
    else:
        cfg = default_cfg

    if overrides:
        override_cfg = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return cfg
