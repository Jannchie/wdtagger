from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
from _typeshed import Incomplete
from PIL import Image

Input = Union[np.ndarray, Image.Image, str, Path]

__all__ = ["Tagger"]

class Result:
    general_tag_data: Incomplete
    character_tag_data: Incomplete
    rating_data: Incomplete
    def __init__(self, pred, sep_tags, general_threshold: float = 0.35, character_threshold: float = 0.9) -> None: ...
    @property
    def general_tags(self): ...
    @property
    def character_tags(self): ...
    @property
    def rating(self): ...
    @property
    def general_tags_string(self) -> str: ...
    @property
    def character_tags_string(self) -> str: ...
    @property
    def all_tags(self) -> list[str]: ...
    @property
    def all_tags_string(self) -> str: ...

class Tagger:
    console: Incomplete
    logger: Incomplete
    model_target_size: Incomplete
    cache_dir: Incomplete
    hf_token: Incomplete
    def __init__(
        self,
        model_repo: str = "SmilingWolf/wd-swinv2-tagger-v3",
        cache_dir: Incomplete | None = None,
        hf_token=...,
        loglevel=...,
        num_threads: Incomplete | None = None,
        providers: Incomplete | None = None,
        console: Incomplete | None = None,
    ) -> None: ...
    sep_tags: Incomplete
    model: Incomplete
    def load_model(
        self,
        model_repo,
        cache_dir: Incomplete | None = None,
        hf_token: Incomplete | None = None,
        num_threads: int | None = None,
        providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
    ): ...
    def pil_to_cv2_numpy(self, image): ...
    def tag(
        self, image: Input | list[Input], general_threshold: float = 0.35, character_threshold: float = 0.9
    ) -> Result | list[Result]: ...
