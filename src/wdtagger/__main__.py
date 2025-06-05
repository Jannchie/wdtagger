import argparse

from PIL import Image

from wdtagger import Tagger


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag images using wdtagger.")
    parser.add_argument("images", nargs="+", help="Path(s) to image file(s) to tag.")
    parser.add_argument("--general-threshold", type=float, default=0.35, help="Threshold for general tags.")
    parser.add_argument("--character-threshold", type=float, default=0.9, help="Threshold for character tags.")
    args = parser.parse_args()

    tagger = Tagger()
    imgs = [Image.open(img_path) for img_path in args.images]
    results = tagger.tag(imgs, general_threshold=args.general_threshold, character_threshold=args.character_threshold)
    if not isinstance(results, list):
        results = [results]
    for i, result in enumerate(results):
        print(f"Image {args.images[i]}:")  # noqa: T201
        print(result)  # noqa: T201
        print()  # noqa: T201


if __name__ == "__main__":
    main()
