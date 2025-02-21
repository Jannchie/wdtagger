from PIL import Image

from wdtagger import Tagger

if __name__ == "__main__":
    tagger = Tagger()
    images = [
        Image.open("./tests/images/赤松楓.9d64b955.jpeg"),
    ]
    results = tagger.tag(images)
