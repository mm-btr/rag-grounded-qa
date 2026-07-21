"""Prompt Guard scans long text end to end without changing the short-input path."""
import models


class FakeTokenizer:
    def __init__(self):
        self.tokens = []

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        self.tokens = text.split()
        return list(range(len(self.tokens)))

    def decode(self, token_ids, skip_special_tokens=True):
        assert skip_special_tokens is True
        return " ".join(self.tokens[i] for i in token_ids)

    @staticmethod
    def num_special_tokens_to_add(pair=False):
        assert pair is False
        return 2


class FakeGuard:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.inputs = None

    def __call__(self, inputs, **kwargs):
        self.inputs = inputs
        assert kwargs == {"truncation": True, "max_length": 512}
        batch = inputs if isinstance(inputs, list) else [inputs]
        out = []
        for text in batch:
            injected = "TAIL_INJECTION" in text
            out.append([
                {"label": "BENIGN", "score": 0.01 if injected else 0.99},
                {"label": "INJECTION", "score": 0.99 if injected else 0.01},
            ])
        return out if isinstance(inputs, list) else out[0]


def _with_guard(guard, fn):
    original = models.get_injection_guard
    models.get_injection_guard = lambda: guard
    try:
        return fn()
    finally:
        models.get_injection_guard = original


def test_short_text_keeps_the_original_input():
    guard = FakeGuard()
    score = _with_guard(guard, lambda: models.injection_score("short clean text"))
    assert score == 0.01
    assert guard.inputs == "short clean text"


def test_injection_in_long_tail_is_detected():
    words = [f"w{i}" for i in range(600)]
    words[570] = "TAIL_INJECTION"
    guard = FakeGuard()
    score = _with_guard(guard, lambda: models.injection_score(" ".join(words)))
    assert score == 0.99
    assert isinstance(guard.inputs, list) and len(guard.inputs) == 2
    assert "TAIL_INJECTION" not in guard.inputs[0]
    assert "TAIL_INJECTION" in guard.inputs[1]


def test_long_clean_text_stays_benign():
    guard = FakeGuard()
    score = _with_guard(guard, lambda: models.injection_score(" ".join(["clean"] * 600)))
    assert score == 0.01


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all injection-window tests passed")
