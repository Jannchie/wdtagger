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
