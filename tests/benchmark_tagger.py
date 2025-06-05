import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from wdtagger import Tagger

tagger = Tagger()
image_dir = Path("./tests/images/")
image_paths = [(image_dir / image) for image in os.listdir(image_dir)] * 16
images = [Image.open(image_path) for image_path in image_paths]


def tag_in_batch(images: Sequence[Image.Image], batch: int = 1) -> None:
    for i in range(0, len(images), batch):
        tagger.tag(images[i : i + batch])


@pytest.mark.benchmark(
    group="tagger",
    min_rounds=10,
    warmup=False,
    disable_gc=True,
)
@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
def test_tagger_benchmark(benchmark: Any, batch: int) -> None:  # noqa: ANN401
    # warmup
    tag_in_batch(images[:1])
    benchmark.pedantic(tag_in_batch, args=(images, batch), iterations=1, rounds=10)


# cmd: pytest tests/benchmark_tagger.py -v
