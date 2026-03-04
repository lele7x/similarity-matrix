
import os
# Avoid VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import psutil
import numpy as np
from typing import Optional
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from torch import Tensor
from torch.nn.functional import normalize as torch_norm

from similarity_matrix.lib.logging import logger
from similarity_matrix.lib.cache import file_cache


# -----------------------------------------------------------------------
# Model

def get_model_size(model, float_bit=32):
    """
    Calculate the estimated size of the SentenceTransformer model based on the number of parameters.

    Args:
    - model: The SentenceTransformer model object.
    - float_bit (int): Number of bits per floating point number.

    Returns:
    - size (int): Estimated size of the model in bytes.
    """
    # Calculate the total number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate the size in bytes (assuming float_bit bits per parameter for
    # floating point numbers)
    size_bytes = (num_params * float_bit) // 8

    return size_bytes


def initialize_model(
        model_name: str = 'jinaai/jina-embeddings-v3',
        use_cuda: bool = True,
        force_sentence_transformer: bool = False,
        **kwargs) -> SentenceTransformer:
    """
    Initialize the SentenceTransformer model with optional CUDA support,
    performing a memory check to ensure sufficient GPU memory.

    Args:
    - model_name (str): The name of the SentenceTransformer model to initialize.
    - use_cuda (bool): Whether to use CUDA for GPU acceleration.

    Returns:
    - model (SentenceTransformer): The initialized SentenceTransformer model.
    """
    from_automodel = ("/" in model_name)
    if use_cuda and torch.cuda.is_available():
        logger.info("Using GPU for model inference.")

        # NOTE: this is to ensure CUDA errors are caught
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # Empty the GPU cache to free up memory
        torch.cuda.empty_cache()

        device = torch.device("cuda")
        if from_automodel and (not force_sentence_transformer):
            model = AutoModel.from_pretrained(
                model_name, device_map='cuda', trust_remote_code=True, **kwargs)
        else:
            model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=True,
                **kwargs)
    else:
        # If the user wanted to use CUDA show a warning that in fact it is
        # not being used
        if use_cuda:
            logger.warning("Warning! Using CPU for model inference.")
        if from_automodel and (not force_sentence_transformer):
            model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True, **kwargs)
        else:
            model = SentenceTransformer(
                model_name, trust_remote_code=True, **kwargs)

    return model


def cos_sim(rows: list | np.ndarray | Tensor,
            columns: list | np.ndarray | Tensor) -> Tensor:
    """
    Computes the cosine similarity between two tensors.

    Args:
        rows (Union[list, np.ndarray, Tensor]): The first tensor (rows).
        columns (Union[list, np.ndarray, Tensor]): The second tensor (columns).

    Returns:
        Tensor: Matrix with res[i][j] = cos_sim(rows[i], columns[j])
    """
    # Convert inputs to Tensor type
    if not isinstance(rows, Tensor):
        rows = torch.tensor(rows)
    if not isinstance(columns, Tensor):
        columns = torch.tensor(columns)

    # If the tensors are 1-dimensional, they are unsqueezed to add a batch
    # dimension
    if rows.dim() == 1:
        rows = rows.unsqueeze(0)
    if columns.dim() == 1:
        columns = columns.unsqueeze(0)

    # Normalize the matrices, so that each sentence embedding has unit length
    rows_norm = torch_norm(rows, p=2, dim=1)
    columns_norm = torch_norm(columns, p=2, dim=1)

    # Compute cosine similarity: normalized_rows @ normalized_columns.T
    return torch.mm(rows_norm, columns_norm.transpose(0, 1))


@file_cache()
def encode(model: SentenceTransformer, texts: list[str]):
    """
    Encode the texts using model.

    This method takes advantage of a filesystem caching
    mechanism to avoid repeated embedding evaulations.
    """
    # Disable gradient tracking for inference
    # This is more efficient and avoids unnecessary memory usage
    # since we don't need to compute gradients during inference.
    with torch.no_grad():
        # Encode the texts using the model
        # convert_to_tensor=True ensures the output is a PyTorch tensor
        # which is more efficient for further processing
        # and avoids unnecessary conversions
        # to numpy arrays.
        return model.encode(texts, convert_to_tensor=True)


def get_memory_capacity(device: Optional[str] = None) -> float:
    """
    Get the available memory capacity in GB.
    Returns VRAM if CUDA is available and being used, otherwise returns RAM.

    Args:
        device: Optional device specification ('cuda', 'cpu', or None for auto-detection)

    Returns:
        float: Available memory capacity in GB
    """
    # Determine which device to check
    if device is None:
        use_cuda = torch.cuda.is_available()
    elif device.lower() == 'cuda':
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            logger.warning(
                "CUDA requested but not available, falling back to CPU/RAM")
    elif device.lower() == 'cpu':
        use_cuda = False
    else:
        raise ValueError(
            f"Invalid device: {device}. Must be 'cuda', 'cpu', or None")

    if use_cuda:
        try:
            # Get VRAM capacity
            device_properties = torch.cuda.get_device_properties(0)
            total_vram_bytes = device_properties.total_memory
            total_vram_gb = total_vram_bytes / (1024**3)

            # Get currently allocated memory to calculate available VRAM
            allocated_vram_bytes = torch.cuda.memory_allocated(0)
            reserved_vram_bytes = torch.cuda.memory_reserved(0)

            # Available VRAM is total minus reserved (reserved includes
            # allocated)
            available_vram_bytes = total_vram_bytes - reserved_vram_bytes
            available_vram_gb = available_vram_bytes / (1024**3)

            logger.info(f"CUDA device: {device_properties.name}")
            logger.info(f"Total VRAM: {total_vram_gb:.2f} GB")
            logger.info(f"Available VRAM: {available_vram_gb:.2f} GB")
            logger.info(
                f"Reserved VRAM: {reserved_vram_bytes / (1024**3):.2f} GB")

            return available_vram_gb

        except Exception as e:
            logger.error(f"Error getting VRAM info: {e}. Falling back to RAM.")
            use_cuda = False

    if not use_cuda:
        # Get RAM capacity
        memory_info = psutil.virtual_memory()
        total_ram_bytes = memory_info.total
        available_ram_bytes = memory_info.available

        total_ram_gb = total_ram_bytes / (1024**3)
        available_ram_gb = available_ram_bytes / (1024**3)

        logger.info(f"Total RAM: {total_ram_gb:.2f} GB")
        logger.info(f"Available RAM: {available_ram_gb:.2f} GB")

        return available_ram_gb


def get_total_memory_capacity(device: Optional[str] = None) -> float:
    """
    Get the total memory capacity in GB (not considering current usage).
    Returns total VRAM if CUDA is available and being used, otherwise returns total RAM.

    Args:
        device: Optional device specification ('cuda', 'cpu', or None for auto-detection)

    Returns:
        float: Total memory capacity in GB
    """
    # Determine which device to check
    if device is None:
        use_cuda = torch.cuda.is_available()
    elif device.lower() == 'cuda':
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            logger.warning(
                "CUDA requested but not available, falling back to CPU/RAM")
    elif device.lower() == 'cpu':
        use_cuda = False
    else:
        raise ValueError(
            f"Invalid device: {device}. Must be 'cuda', 'cpu', or None")

    if use_cuda:
        try:
            # Get total VRAM capacity
            device_properties = torch.cuda.get_device_properties(0)
            total_vram_bytes = device_properties.total_memory
            total_vram_gb = total_vram_bytes / (1024**3)

            logger.info(f"CUDA device: {device_properties.name}")
            logger.info(f"Total VRAM: {total_vram_gb:.2f} GB")

            return total_vram_gb

        except Exception as e:
            logger.error(f"Error getting VRAM info: {e}. Falling back to RAM.")
            use_cuda = False

    if not use_cuda:
        # Get total RAM capacity
        memory_info = psutil.virtual_memory()
        total_ram_bytes = memory_info.total
        total_ram_gb = total_ram_bytes / (1024**3)

        logger.info(f"Total RAM: {total_ram_gb:.2f} GB")

        return total_ram_gb
