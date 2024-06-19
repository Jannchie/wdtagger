import os

import pytest
from PIL import Image

from wdtagger import Tagger

tagger = Tagger()
image_dir = "./tests/images/"
image_paths = [os.path.join(image_dir, image) for image in os.listdir(image_dir)] * 16
images = [Image.open(image_path) for image_path in image_paths]


def tag_in_batch(images, batch=1):
    for i in range(0, len(images), batch):
        tagger.tag(images[i : i + batch])


@pytest.mark.benchmark(
    group="tagger",
    min_rounds=10,
    warmup=False,
    disable_gc=True,
)
@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
def test_tagger_benchmark(benchmark, batch):
    # warmup
    tag_in_batch(images[:1])
    benchmark.pedantic(tag_in_batch, args=(images, batch), iterations=1, rounds=10)


# cmd: pytest tests/benchmark_tagger.py -v
