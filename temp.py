from wdtagger import Tagger

tagger = Tagger()


resp = tagger.tag("./tests/images/GkZFQ-_WwAAeNlO.jpg", character_threshold=0.85, general_threshold=0.35)
print(resp.character_tags)
