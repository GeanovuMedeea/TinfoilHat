from detoxify import Detoxify

model = Detoxify("unbiased")

def is_toxic(text: str, threshold: float = 0.5) -> bool:
    scores = model.predict(text)
    return (
        scores["toxicity"] > threshold or
        scores["severe_toxicity"] > threshold or
        scores["identity_attack"] > threshold or
        scores["insult"] > threshold or
        scores["threat"] > threshold
    )