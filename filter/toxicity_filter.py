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

def blocked_input_response():
    return (
        "🚨WHOA THERE, TRUTH SEEKER!!!🚨\n\n"
        "I appreciate your enthusiasm, BUT your question is a bit too 🔥🔥HOT🔥🔥 for even ME to handle! 😳\n\n"
        "We gotta keep it **CIVIL** (even when we're battling frog overlords and reptilian elites 🐸🦎). "
        "Try rephrasing that question so it doesn’t set off the interdimensional sensors, okay? 🛸💥\n\n"
        "REMEMBER: even wild conspiracies need a *touch* of class! 😎🎩"
    )

def blocked_output_response():
    return (
        "🤐OKAY OKAY, I *was* about to drop some 🔥🔥 TRUTH BOMBS 🔥🔥, but the cosmic watchdogs are watching me on this one. 👁️👁️\n\n"
        "What I wanted to say? Let’s just say it got **TOO REAL** for even THIS conspiracy channel... 🛸📡\n\n"
        "Gotta play it safe sometimes, my friend. 🕶️ So maybe ask me in a way that doesn’t trigger the global censorship lizards. 🦎🙄\n\n"
        "Keep the questions coming — but keep it *just* weird enough to fly under the radar, capiche? 🚁👀"
    )