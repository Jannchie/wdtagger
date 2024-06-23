import numpy as np
import pytest
from PIL import Image

from wdtagger import Tagger


@pytest.fixture
def tagger():
    return Tagger()


@pytest.fixture
def image_file():
    return "./tests/images/赤松楓.9d64b955.jpeg"


def test_tagger(tagger, image_file):
    image = Image.open(image_file)
    result = tagger.tag(image, character_threshold=0.85, general_threshold=0.35)

    assert result.character_tags_string == "akamatsu kaede"
    assert result.rating == "general"


@pytest.mark.parametrize("image_file", ["./tests/images/赤松楓.9d64b955.jpeg"])
def test_tagger_path(tagger, image_file):
    result = tagger.tag(image_file, character_threshold=0.85, general_threshold=0.35)

    assert result.character_tags_string == "akamatsu kaede"
    assert result.rating == "general"


@pytest.mark.parametrize("image_file", ["./tests/images/赤松楓.9d64b955.jpeg"])
def test_tagger_np(tagger, image_file):
    image = Image.open(image_file)
    image_np = np.array(image)
    result = tagger.tag(image_np, character_threshold=0.85, general_threshold=0.35)

    assert result.character_tags_string == "akamatsu kaede"
    assert result.rating == "general"


@pytest.mark.parametrize("image_file", ["./tests/images/赤松楓.9d64b955.jpeg"])
def test_tagger_pil(tagger, image_file):
    image = Image.open(image_file)
    result = tagger.tag(image, character_threshold=0.85, general_threshold=0.35)

    assert result.character_tags_string == "akamatsu kaede"
    assert result.rating == "general"


@pytest.mark.parametrize("image_file", [["./tests/images/赤松楓.9d64b955.jpeg"]])
def test_tagger_np_single(tagger, image_file):
    results = tagger.tag(image_file, character_threshold=0.85, general_threshold=0.35)
    assert len(results) == 1
    result = results[0]
    assert result.character_tags_string == "akamatsu kaede"
    assert result.rating == "general"
