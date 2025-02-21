import numpy as np
import pytest
from PIL import Image

from wdtagger import Tagger


@pytest.fixture
def tagger() -> Tagger:
    """
    Create and return a new instance of the Tagger class.

    Returns:
        Tagger: An instance of the Tagger class.
    """
    return Tagger()


@pytest.fixture
def image_file() -> str:
    return "./tests/images/赤松楓.9d64b955.jpeg"


def test_tagger(tagger: Tagger, image_file: str) -> None:
    image = Image.open(image_file)
    result = tagger.tag(image, character_threshold=0.85, general_threshold=0.35)

    assert result.character_tags_string == "akamatsu_kaede"
    assert result.rating == "general"


@pytest.mark.parametrize("image_file", ["./tests/images/赤松楓.9d64b955.jpeg"])
def test_tagger_path_single(tagger: Tagger, image_file: str) -> None:
    result = tagger.tag(image_file, character_threshold=0.85, general_threshold=0.35)

    assert result.character_tags_string == "akamatsu_kaede"
    assert result.rating == "general"


@pytest.mark.parametrize("image_file", ["./tests/images/赤松楓.9d64b955.jpeg"])
def test_tagger_np(tagger: Tagger, image_file: str) -> None:
    image = Image.open(image_file)
    image_np = np.array(image)
    result = tagger.tag(image_np, character_threshold=0.85, general_threshold=0.35)

    assert result.character_tags_string == "akamatsu_kaede"
    assert result.rating == "general"


@pytest.mark.parametrize("image_file", ["./tests/images/赤松楓.9d64b955.jpeg"])
def test_tagger_pil(tagger: Tagger, image_file: str) -> None:
    image = Image.open(image_file)
    result = tagger.tag(image, character_threshold=0.85, general_threshold=0.35)

    assert result.character_tags_string == "akamatsu_kaede"
    assert result.rating == "general"


@pytest.mark.parametrize("image_file", ["./tests/images/赤松楓.9d64b955.jpeg"])
def test_tagger_path_multi(tagger: Tagger, image_file: str) -> None:
    results = tagger.tag([image_file, image_file], character_threshold=0.85, general_threshold=0.35)
    assert len(results) == 2
    result = results[0]
    assert result.character_tags_string == "akamatsu_kaede"
    assert result.rating == "general"
