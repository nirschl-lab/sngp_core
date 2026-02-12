"""
Utilities for uploading and downloading SNGP/Baseline models from Hugging Face Hub.
Handles auto-download and inference without requiring Transformers compatibility.
"""

import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
from loguru import logger

try:
    from huggingface_hub import hf_hub_download, HfApi, CommitInfo
    from huggingface_hub.utils import RepositoryNotFoundError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("huggingface_hub not installed. Install with: pip install huggingface_hub")


class SNGPModelConfig:
    """Configuration object for SNGP models."""
    def __init__(self, **kwargs):
        self.arch: str = kwargs.get("arch", "resnet18")
        self.num_classes: int = kwargs.get("num_classes", 4)
        self.rff_dim: int = kwargs.get("rff_dim", 1024)
        self.length_scale: float = kwargs.get("length_scale", 1.0)
        self.ridge_penalty: float = kwargs.get("ridge_penalty", 1e-3)
        self.cov_momentum: float = kwargs.get("cov_momentum", 0.999)
        self.mean_field: bool = kwargs.get("mean_field", True)
        self.pretrained_backbone: bool = kwargs.get("pretrained_backbone", False)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SNGPModelConfig":
        return cls(**config_dict)


class BaselineModelConfig:
    """Configuration object for Baseline models."""
    def __init__(self, **kwargs):
        self.arch: str = kwargs.get("arch", "resnet18")
        self.num_classes: int = kwargs.get("num_classes", 4)
        self.dropout_p: float = kwargs.get("dropout_p", 0.5)
        self.pretrained: bool = kwargs.get("pretrained", False)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaselineModelConfig":
        return cls(**config_dict)


class HFModelLoader:
    """
    Unified loader for SNGP and Baseline models from Hugging Face Hub.
    
    Handles:
    - Auto-downloading checkpoints from HF Hub
    - Loading model architecture and weights
    - Caching locally for repeated use
    
    Usage:
        loader = HFModelLoader(repo_id="nirschl-lab/sngp-models")
        model, config = loader.load_model("wong_sngp_resnet18")
        predictions = model(input_tensor)
    """
    
    def __init__(
        self,
        repo_id: str = "nirschl-lab/sngp-models",
        cache_dir: Optional[str] = None,
    ):
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub required. Install: pip install huggingface_hub")
        
        self.repo_id = repo_id
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "sngp_models")
        self.api = HfApi()
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"HF Model Loader initialized for repo: {repo_id}")
        logger.info(f"Cache directory: {self.cache_dir}")

    def load_model(
        self,
        model_name: str,
        device: Optional[Union[str, torch.device]] = None,
        map_location: Optional[torch.device] = None,
    ) -> Tuple[nn.Module, Union[SNGPModelConfig, BaselineModelConfig]]:
        """
        Load a model and its configuration from HF Hub.
        
        Args:
            model_name: Name of the model (e.g., "wong_sngp_resnet18")
            device: Device to load model to (e.g., "cuda", "cpu")
            map_location: Legacy parameter, use device instead
            
        Returns:
            Tuple of (model, config)
        """
        device = device or (map_location if map_location else ("cuda" if torch.cuda.is_available() else "cpu"))
        if isinstance(device, str):
            device = torch.device(device)
        
        logger.info(f"Loading model: {model_name}")
        
        # Download config
        config_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=f"{model_name}/config.json",
            cache_dir=self.cache_dir,
            repo_type="model",
        )
        logger.info(f"Downloaded config from {config_path}")
        
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        
        model_type = config_dict.pop("model_type")
        
        # Download checkpoint
        ckpt_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=f"{model_name}/model.pt",
            cache_dir=self.cache_dir,
            repo_type="model",
        )
        logger.info(f"Downloaded checkpoint from {ckpt_path}")
        
        # Instantiate model based on type
        if model_type == "sngp":
            from src.models.sngp.sngp_classifier import SNGPClassifier
            config = SNGPModelConfig.from_dict(config_dict)
            model = SNGPClassifier(**config.to_dict())
        elif model_type == "baseline":
            from src.models.baseline.baseline_models import BaselineClassifier
            config = BaselineModelConfig.from_dict(config_dict)
            model = BaselineClassifier(**config.to_dict())
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load weights
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        logger.info(f"Model loaded successfully on {device}")
        
        return model, config

    @staticmethod
    def list_available_models(repo_id: str = "nirschl-lab/sngp-models") -> list:
        """List all available models in the HF Hub repo."""
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub required")
        
        api = HfApi()
        try:
            repo_info = api.repo_info(repo_id, repo_type="model")
            # Parse folder structure - assumes models are in subdirectories
            files = api.list_repo_files(repo_id, repo_type="model")
            
            models = set()
            for file in files:
                if "/" in file:
                    model_name = file.split("/")[0]
                    if model_name and not model_name.startswith("."):
                        models.add(model_name)
            
            return sorted(list(models))
        except RepositoryNotFoundError:
            logger.warning(f"Repository {repo_id} not found")
            return []


class HFModelUploader:
    """
    Utility for uploading SNGP and Baseline models to Hugging Face Hub.
    
    Usage:
        uploader = HFModelUploader(repo_id="nirschl-lab/sngp-models")
        uploader.upload_model(
            model=sngp_model,
            model_name="wong_sngp_resnet18",
            config=config,
            config_dict={"num_classes": 4, "arch": "resnet18", ...}
        )
    """
    
    def __init__(self, repo_id: str = "nirschl-lab/sngp-models"):
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub required. Install: pip install huggingface_hub")
        
        self.repo_id = repo_id
        self.api = HfApi()
        logger.info(f"HF Model Uploader initialized for repo: {repo_id}")

    def upload_model(
        self,
        model: nn.Module,
        model_name: str,
        model_type: str,
        config_dict: Dict[str, Any],
        commit_message: Optional[str] = None,
    ) -> CommitInfo:
        """
        Upload a model and its config to HF Hub.
        
        Args:
            model: PyTorch model to upload
            model_name: Name for the model directory (e.g., "wong_sngp_resnet18")
            model_type: Type of model ("sngp" or "baseline")
            config_dict: Configuration dictionary
            commit_message: Custom commit message
            
        Returns:
            CommitInfo object from HF API
        """
        if model_type not in ["sngp", "baseline"]:
            raise ValueError(f"model_type must be 'sngp' or 'baseline', got {model_type}")
        
        commit_message = commit_message or f"Upload {model_name}"
        
        # Create temporary directory for upload
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model state dict
            model_path = os.path.join(tmpdir, "model.pt")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved model to {model_path}")
            
            # Save config
            config_dict["model_type"] = model_type
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Saved config to {config_path}")
            
            # Upload
            commit_info = self.api.upload_folder(
                repo_id=self.repo_id,
                folder_path=tmpdir,
                path_in_repo=model_name,
                commit_message=commit_message,
                repo_type="model",
            )
            logger.info(f"Uploaded {model_name} to {self.repo_id}")
        
        return commit_info


def quick_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    model_type: str = "baseline",
    device: Optional[torch.device] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Quick inference wrapper that handles both model types.
    
    Args:
        model: Loaded model
        input_tensor: Input batch
        model_type: "baseline" or "sngp"
        device: Device to run on
        
    Returns:
        For baseline: logits [B, num_classes]
        For sngp: (mean_field_logits, raw_logits, pred_var)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        if model_type == "sngp":
            mean_field_logits, raw_logits, pred_var = model(input_tensor)
            return mean_field_logits, raw_logits, pred_var
        else:  # baseline
            logits = model(input_tensor)
            return logits
