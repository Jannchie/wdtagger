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
