import logging
import os
import time
from collections import OrderedDict

import huggingface_hub
import numpy as np
import onnxruntime as rt
import pandas as pd
import rich
import rich.live
from PIL import Image
from rich.logging import RichHandler

# Access console for rich text and logging
console = rich.get_console()

# Environment variables and file paths
HF_TOKEN = os.environ.get(
    "HF_TOKEN", ""
)  # Token for authentication with HuggingFace API
MODEL_FILENAME = "model.onnx"  # ONNX model filename
LABEL_FILENAME = "selected_tags.csv"  # Labels CSV filename


def load_labels(dataframe) -> list[str]:
    """Load labels from a dataframe and process tag names.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the tag names and categories.

    Returns:
        tag_names: List of tag names.
        rating_indexes: List of indexes for rating tags.
        general_indexes: List of indexes for general tags.
        character_indexes: List of indexes for character tags.
    """
    name_series = dataframe["name"]
    kaomojis = [
        "0_0",
        "(o)_(o)",
        "+_+",
        "+_-",
        "._.",
        "<o>_<o>",
        "<|>_<|>",
        "=_=",
        ">_<",
        "3_3",
        "6_9",
        ">_o",
        "@_@",
        "^_^",
        "o_o",
        "u_u",
        "x_x",
        "|_|",
        "||_||",
    ]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    )
    tag_names = name_series.tolist()
    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])

    return tag_names, rating_indexes, general_indexes, character_indexes


class Result:
    def __init__(
        self,
        pred,
        sep_tags,
        general_threshold=0.35,
        character_threshold=0.9,
    ):
        """Initialize the Result object to store tagging results.

        Args:
            preds (np.array): Predictions array from the model.
            sep_tags (tuple): Tuple containing separated tags based on categories.
            general_threshold (float): Threshold for general tags.
            character_threshold (float): Threshold for character tags.
        """
        tag_names = sep_tags[0]
        rating_indexes = sep_tags[1]
        general_indexes = sep_tags[2]
        character_indexes = sep_tags[3]
        labels = list(zip(tag_names, pred.astype(float)))

        # Ratings
        ratings_names = [labels[i] for i in rating_indexes]
        rating_data = dict(ratings_names)
        rating_data = OrderedDict(
            sorted(rating_data.items(), key=lambda x: x[1], reverse=True)
        )

        # General tags
        general_names = [labels[i] for i in general_indexes]
        general_tag = [x for x in general_names if x[1] > general_threshold]
        general_tag = OrderedDict(sorted(general_tag, key=lambda x: x[1], reverse=True))

        # Character tags
        character_names = [labels[i] for i in character_indexes]
        character_tag = [x for x in character_names if x[1] > character_threshold]
        character_tag = OrderedDict(
            sorted(character_tag, key=lambda x: x[1], reverse=True)
        )

        self.general_tag_data = general_tag
        self.character_tag_data = character_tag
        self.rating_data = rating_data

    @property
    def general_tags(self):
        """Return general tags as a tuple."""
        return tuple(self.general_tag_data.keys())

    @property
    def character_tags(self):
        """Return character tags as a tuple."""
        return tuple(self.character_tag_data.keys())

    @property
    def rating(self):
        """Return the highest rated tag."""
        return max(self.rating_data, key=self.rating_data.get)

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

    def __str__(self) -> str:
        """Return a formatted string representation of the tags and their ratings."""

        def get_tag_with_rate(tag_dict):
            return ", ".join([f"{k} ({v:.2f})" for k, v in tag_dict.items()])

        result = f"General tags: {get_tag_with_rate(self.general_tag_data)}\n"
        result += f"Character tags: {get_tag_with_rate(self.character_tag_data)}\n"
        result += f"Rating: {self.rating} ({self.rating_data[self.rating]:.2f})"
        return result


class Tagger:
    def __init__(
        self,
        model_repo="SmilingWolf/wd-swinv2-tagger-v3",
        cache_dir=None,
        hf_token=HF_TOKEN,
        loglevel=logging.INFO,
    ):
        """Initialize the Tagger object with the model repository and tokens.

        Args:
            model_repo (str): Repository name on HuggingFace.
            cache_dir (str, optional): Directory to cache the model. Defaults to None.
            hf_token (str, optional): HuggingFace token for authentication. Defaults to HF_TOKEN.
        """
        self.logger = logging.getLogger("wdtagger")
        self.logger.setLevel(loglevel)
        self.logger.addHandler(RichHandler())
        self.model_target_size = None
        self.cache_dir = cache_dir
        self.hf_token = hf_token
        self.load_model(model_repo, cache_dir, hf_token)

    def load_model(
        self, model_repo, cache_dir=None, hf_token=None, num_threads: int = None
    ):
        """Load the model and tags from the specified repository.

        Args:
            model_repo (str): Repository name on HuggingFace.
            cache_dir (str, optional): Directory to cache the model. Defaults to None.
            hf_token (str, optional): HuggingFace token for authentication. Defaults to None.
        """
        with console.status("Loading model..."):
            csv_path = huggingface_hub.hf_hub_download(
                model_repo,
                LABEL_FILENAME,
                cache_dir=cache_dir,
                use_auth_token=hf_token,
            )

            model_path = huggingface_hub.hf_hub_download(
                model_repo,
                MODEL_FILENAME,
                cache_dir=cache_dir,
                use_auth_token=hf_token,
            )

            tags_df = pd.read_csv(csv_path)
            self.sep_tags = load_labels(tags_df)
            options = rt.SessionOptions()
            if num_threads:
                options.intra_op_num_threads = num_threads
                options.inter_op_num_threads = num_threads
            model = rt.InferenceSession(model_path, options)
            _, height, _, _ = model.get_inputs()[0].shape
            self.model_target_size = height
            self.model = model

    def prepare_image(self, image):
        """Prepare the image for model input.

        Args:
            image (PIL.Image): Input image.

        Returns:
            np.array: Processed image as a NumPy array.
        """
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        target_size = self.model_target_size
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        # Pad image to square
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC,
            )

        return np.asarray(padded_image, dtype=np.float32)

    def tag(
        self,
        image: Image.Image | list[Image.Image],
        general_threshold=0.35,
        character_threshold=0.9,
    ) -> Result | list[Result]:
        """Tag the image and return the results.

        Args:
            image (PIL.Image | list[PIL.Image]): Input image or list of images.
            general_threshold (float): Threshold for general tags.
            character_threshold (float): Threshold for character tags.

        Returns:
            Result | list[Result]: Tagging results.
        """
        started_at = time.time()
        images = [image] if isinstance(image, Image.Image) else image
        images = [self.prepare_image(img) for img in images]
        image_array = np.asarray(images, dtype=np.float32)
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image_array})[0]
        results = [
            Result(pred, self.sep_tags, general_threshold, character_threshold)
            for pred in preds
        ]
        duration = time.time() - started_at
        image_length = len(images)
        self.logger.info(
            f"Tagging {image_length} image{ 's' if image_length > 1 else ''} took {duration:.2f} seconds."
        )
        return results[0] if len(results) == 1 else results


__all__ = ["Tagger"]

if __name__ == "__main__":
    tagger = Tagger()
    image = Image.open("./tests/images/赤松楓.9d64b955.jpeg")
    result = tagger.tag(image)
    tagger.logger.info(result)
