import importlib.resources
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, overload

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from PIL.ImageFile import ImageFile
from timm.data import create_transform, resolve_data_config
from torch import Tensor, nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from torchvision.transforms import Compose


HF_TOKEN = os.environ.get("HF_TOKEN", "")


Input = np.ndarray | Image.Image | str | Path | ImageFile


@dataclass
class LabelData:
    names: list[str]
    rating: list[np.int64]
    general: list[np.int64]
    character: list[np.int64]


def load_labels() -> LabelData:
    file = importlib.resources.as_file(importlib.resources.files("wdtagger.assets").joinpath("selected_tags.csv"))
    with file as f:
        df: pd.DataFrame = pd.read_csv(f, usecols=["name", "category"])
    rating_catagory_idx = 9
    general_catagory_idx = 0
    character_catagory_idx = 4
    return LabelData(
        names=df["name"].tolist(),
        rating=list(np.where(df["category"] == rating_catagory_idx)[0]),
        general=list(np.where(df["category"] == general_catagory_idx)[0]),
        character=list(np.where(df["category"] == character_catagory_idx)[0]),
    )


def pil_ensure_rgb(image: Image.Image) -> Image.Image:
    # convert to RGB/RGBA if not already (deals with palette images etc.)
    if image.mode not in ["RGB", "RGBA"]:
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    # convert RGBA to RGB with white background
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def pil_pad_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    # get the largest dimension so we can pad to a square
    px = max(image.size)
    # pad to square with white background
    canvas = Image.new("RGB", (px, px), (255, 255, 255))
    canvas.paste(image, ((px - w) // 2, (px - h) // 2))
    return canvas


def to_pil(img: Input) -> Image.Image:
    if isinstance(img, str | Path):
        return Image.open(img)
    if isinstance(img, np.ndarray):
        return Image.fromarray(img)
    if isinstance(img, Image.Image):
        return img
    msg = "Invalid input type."
    raise ValueError(msg)


def get_tags(
    probs: Tensor,
    labels: LabelData,
    gen_threshold: float,
    char_threshold: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    # Convert indices+probs to labels
    probs_list = list(zip(labels.names, probs.numpy(), strict=False))

    # First 4 labels are actually ratings
    rating_labels = dict([probs_list[i] for i in labels.rating])

    # General labels, pick any where prediction confidence > threshold
    gen_labels = [probs_list[i] for i in labels.general]
    gen_labels = dict([x for x in gen_labels if x[1] > gen_threshold])
    gen_labels = dict(sorted(gen_labels.items(), key=lambda item: item[1], reverse=True))

    # Character labels, pick any where prediction confidence > threshold
    char_labels = [probs_list[i] for i in labels.character]
    char_labels = dict([x for x in char_labels if x[1] > char_threshold])
    char_labels = dict(sorted(char_labels.items(), key=lambda item: item[1], reverse=True))

    return rating_labels, char_labels, gen_labels


class Result:
    def __init__(
        self,
        rating_data: dict[str, float],
        character_tag: dict[str, float],
        general_tag: dict[str, float],
    ) -> None:
        """Initialize the Result object with the tags and their ratings."""
        self.general_tag_data = general_tag
        self.character_tag_data = character_tag
        self.rating_data = rating_data

    @property
    def general_tags(self) -> tuple[str, ...]:
        """Return general tags as a tuple."""

        return tuple(
            d.replace("_", " ")
            for d in sorted(
                self.general_tag_data,
                key=lambda k: self.general_tag_data[k],
                reverse=True,
            )
        )

    @property
    def character_tags(self) -> tuple[str, ...]:
        """Return character tags as a tuple."""
        return tuple(
            d.replace("_", " ")
            for d in sorted(
                self.character_tag_data,
                key=lambda k: self.character_tag_data[k],
                reverse=True,
            )
        )

    @property
    def rating(self) -> Literal["general", "sensitive", "questionable", "explicit"]:
        """Return the highest rated tag."""
        return max(self.rating_data, key=lambda k: self.rating_data[k])  # type: ignore

    @property
    def general_tags_string(self) -> str:
        """Return general tags as a sorted string."""
        string = sorted(
            self.general_tag_data.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        string = [x[0] for x in string]
        return ", ".join(string)

    @property
    def character_tags_string(self) -> str:
        """Return character tags as a sorted string."""
        string = sorted(
            self.character_tag_data.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        string = [x[0] for x in string]
        return ", ".join(string)

    @property
    def all_tags(self) -> list[str]:
        """Return all tags as a list."""
        return [self.rating, *list(self.character_tags), *list(self.general_tags)]

    @property
    def all_tags_string(self) -> str:
        return ", ".join(self.all_tags)

    def __str__(self) -> str:
        """Return a formatted string representation of the tags and their ratings."""

        def get_tag_with_rate(tag_dict: dict[str, float]) -> str:
            return ", ".join([f"{k} ({v:.2f})" for k, v in tag_dict.items()])

        result = f"General tags: {get_tag_with_rate(self.general_tag_data)}\n"
        result += f"Character tags: {get_tag_with_rate(self.character_tag_data)}\n"
        result += f"Rating: {self.rating} ({self.rating_data[self.rating]:.2f})"
        return result


class Tagger:
    def __init__(
        self,
        model_repo: str = "SmilingWolf/wd-swinv2-tagger-v3",
        hf_token: str = HF_TOKEN,
    ) -> None:
        """Initialize the Tagger object with the model repository and tokens."""
        self.logger = logging.getLogger("wdtagger")
        self.model_target_size = None
        self.hf_token = hf_token
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(model_repo)

    def _load_model(
        self,
        model_repo: str,
    ) -> None:
        """Load the model and tags from the specified repository.

        Args:
            model_repo (str): Repository name on HuggingFace.
            cache_dir (str, optional): Directory to cache the model. Defaults to None.
            hf_token (str, optional): HuggingFace token for authentication. Defaults to None.
            num_threads (int, optional): Number of threads for ONNX runtime. Defaults to None.
        """
        start_time = time.time()
        self.logger.info("Loading model from %s", model_repo)
        self._do_load_model(model_repo)
        self.logger.info("Model loaded successfully in %.2fs", time.time() - start_time)

    def _do_load_model(self, model_repo: str) -> None:
        model: nn.Module = timm.create_model(f"hf-hub:{model_repo}").eval()
        state_dict = timm.models.load_state_dict_from_hf(model_repo)
        model.load_state_dict(state_dict)

        self.labels: LabelData = load_labels()

        self.transform: Compose = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))  # type: ignore

        self.model = model.to(self.torch_device)

    @overload
    def tag(
        self,
        image: Input,
        general_threshold: float = 0.35,
        character_threshold: float = 0.9,
    ) -> Result: ...

    @overload
    def tag(
        self,
        image: Sequence[Input],
        general_threshold: float = 0.35,
        character_threshold: float = 0.9,
    ) -> Sequence[Result]: ...

    def tag(
        self,
        image: Input | Sequence[Input],
        general_threshold=0.35,
        character_threshold=0.9,
    ) -> Result | Sequence[Result]:
        """Tag the image and return the results.

        Args:
            image (Union[Input, List[Input]]): Input image or list of images to tag.
            general_threshold (float): Threshold for general tags.
            character_threshold (float): Threshold for character tags.

        Returns:
            Result | list[Result]: Tagging results.
        """
        started_at = time.time()
        images = list(image) if isinstance(image, Sequence) and not isinstance(image, str) else [image]
        images = [to_pil(img) for img in images]
        images = [pil_ensure_rgb(img) for img in images]
        images = [pil_pad_square(img) for img in images]
        inputs: Tensor = torch.stack([self.transform(img) for img in images])  # type: ignore
        inputs = inputs[:, [2, 1, 0]]  # BGR to RGB

        with torch.inference_mode():
            # move model to GPU, if available
            if self.torch_device.type != "cpu":
                inputs = inputs.to(self.torch_device)
            # run the model
            outputs = self.model.forward(inputs)
            # apply the final activation function (timm doesn't support doing this internally)
            outputs = F.sigmoid(outputs)
            # move inputs, outputs, and model back to to cpu if we were on GPU
            if self.torch_device.type != "cpu":
                inputs = inputs.to("cpu")
                outputs = outputs.to("cpu")

        results = [
            Result(
                *get_tags(
                    probs=o,
                    labels=self.labels,
                    gen_threshold=general_threshold,
                    char_threshold=character_threshold,
                ),
            )
            for o in outputs
        ]

        duration = time.time() - started_at
        image_length = len(images)
        self.logger.info(
            "Tagged %d image%s in %.2fs",
            image_length,
            "s" if image_length > 1 else "",
            duration,
        )
        if isinstance(image, Sequence) and not isinstance(image, str):
            return results
        return results[0] if len(results) == 1 else results


__all__ = ["Tagger"]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("wdtagger")
    tagger = Tagger()
    images = [
        Image.open("./tests/images/赤松楓.9d64b955.jpeg"),
    ]
    results = tagger.tag(images)
    for result in results:
        logger.info(result.all_tags_string)
