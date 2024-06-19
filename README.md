# wdtagger

`wdtagger` is a simple and easy-to-use wrapper for the tagger model created by [SmilingWolf](https://github.com/SmilingWolf) which is specifically designed for tagging anime illustrations.

## Installation

You can install `wdtagger` via pip:

```bash
pip install wdtagger
```

## Usage

Below is a basic example of how to use wdtagger in your project:

```python
from PIL import Image
from wdtagger import Tagger

tagger = Tagger() # You can provide the model_repo, the default is "SmilingWolf/wd-swinv2-tagger-v3"
image = Image.open("image.jpg")
result = tagger.tag(image)
print(result)
```

You can input a image list to the tagger to use batch processing, it is faster than single image processing (test on RTX 3090):

```log
---------------------------------------------------------------------------------- benchmark 'tagger': 5 tests -----------------------------------------------------------------------------------
Name (time in ms)                  Min                 Max                Mean             StdDev              Median                IQR            Outliers     OPS            Rounds  Iterations
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_tagger_benchmark[16]     540.8711 (1.0)      598.5156 (1.04)     558.2777 (1.0)      22.2954 (4.10)     549.9650 (1.0)      21.7318 (2.51)          2;2  1.7912 (1.0)          10           1
test_tagger_benchmark[8]      558.9445 (1.03)     576.7220 (1.0)      567.9235 (1.02)      5.4381 (1.0)      568.7336 (1.03)      8.6569 (1.0)           2;0  1.7608 (0.98)         10           1
test_tagger_benchmark[4]      590.6479 (1.09)     626.7126 (1.09)     597.9712 (1.07)     11.0124 (2.03)     594.5067 (1.08)     10.7656 (1.24)          1;1  1.6723 (0.93)         10           1
test_tagger_benchmark[2]      622.8689 (1.15)     643.5122 (1.12)     630.1096 (1.13)      7.2365 (1.33)     627.1716 (1.14)      9.5823 (1.11)          3;0  1.5870 (0.89)         10           1
test_tagger_benchmark[1]      700.6986 (1.30)     816.3089 (1.42)     721.7431 (1.29)     33.9031 (6.23)     712.6850 (1.30)     12.8756 (1.49)          1;1  1.3855 (0.77)         10           1
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
