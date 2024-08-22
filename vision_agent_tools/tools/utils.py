import os
import wget
import gdown
import os.path as osp


CURRENT_DIR = osp.dirname(osp.abspath(__file__))
CHECKPOINT_DIR = osp.join(CURRENT_DIR, "checkpoints")
from vision_agent_tools.shared_types import CachePath
from huggingface_hub import snapshot_download


DEFAULT_HF_CHACHE_DIR = "/root/.cache/huggingface/hub"


def download(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        if url.startswith("https://drive.google.com"):
            gdown.download(url, path, quiet=False, fuzzy=True)
        else:
            wget.download(url, out=path)
    return path


def manage_hf_model_cache(hf_model_name: str, cache_dir: CachePath) -> str:
    """
    Helper function to download HuggingFace models and cache them.
    If there is no cache_dir provided then, we use the default HF cache dir location:
    /root/.cache/huggingface/hub/

    Args:
        hf_model_name (str): The HuggingFace model name.
        cache_dir (str | Path | None): The path to the folder where you plan to store the cache.

    Returns:
        str: The path to the location where the files were cached

    """
    model_dir = (
        f"models--{os.path.dirname(hf_model_name)}"
        + f"--{os.path.basename(hf_model_name)}"
    )
    default_cache_model_dir = os.path.join(DEFAULT_HF_CHACHE_DIR, model_dir)

    user_cached_folder = (
        os.path.join(cache_dir, model_dir) if cache_dir is not None else ""
    )

    is_user_cached_folder = True if os.path.exists(user_cached_folder) else False
    is_default_cached_folder = (
        True if os.path.exists(default_cache_model_dir) else False
    )
    is_model_cached = (cache_dir is not None and is_user_cached_folder) or (
        cache_dir is None and is_default_cached_folder
    )
    print("Is the model cached?:  ", is_model_cached)
    print(
        "Using cache folder: ",
        default_cache_model_dir
        if is_default_cached_folder
        else user_cached_folder or cache_dir,
    )
    model_snapshot = snapshot_download(
        hf_model_name,
        cache_dir=cache_dir,
        local_files_only=is_model_cached,
    )
    print("Model_snapshot_path: ", model_snapshot)
    return model_snapshot
