import re


_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for row, ref_token in enumerate(reference, 1):
        current = [row]
        for col, hyp_token in enumerate(hypothesis, 1):
            substitution = previous[col - 1] + (ref_token != hyp_token)
            current.append(min(
                substitution,
                previous[col] + 1,
                current[col - 1] + 1,
            ))
        previous = current
    return previous[-1]


def mixed_tokens(text):
    """Chinese characters are tokens; Latin spans are whitespace words."""
    text = text.lower().strip()
    parts = re.split(r"([\u3400-\u4dbf\u4e00-\u9fff])", text)
    tokens = []
    for part in parts:
        if not part or not part.strip():
            continue
        if _CJK.fullmatch(part):
            tokens.append(part)
        else:
            tokens.extend(re.findall(r"[a-z0-9']+|[^\s]", part))
    return tokens


def char_tokens(text):
    return [char for char in re.sub(r"\s+", "", text.lower())]


class ErrorRateStats:
    def __init__(self):
        self.mixed_errors = 0
        self.mixed_reference = 0
        self.zh_errors = 0
        self.zh_reference = 0
        self.en_errors = 0
        self.en_reference = 0
        self.sentences = 0

    def add(self, reference, hypothesis):
        ref_mixed = mixed_tokens(reference)
        hyp_mixed = mixed_tokens(hypothesis)
        self.mixed_errors += edit_distance(ref_mixed, hyp_mixed)
        self.mixed_reference += len(ref_mixed)
        if _CJK.search(reference):
            ref_chars = char_tokens(reference)
            hyp_chars = char_tokens(hypothesis)
            self.zh_errors += edit_distance(ref_chars, hyp_chars)
            self.zh_reference += len(ref_chars)
        else:
            ref_words = reference.lower().split()
            hyp_words = hypothesis.lower().split()
            self.en_errors += edit_distance(ref_words, hyp_words)
            self.en_reference += len(ref_words)
        self.sentences += 1

    @staticmethod
    def _rate(errors, reference):
        return errors / max(reference, 1)

    def as_dict(self):
        return {
            "mer": self._rate(self.mixed_errors, self.mixed_reference),
            "cer": self._rate(self.zh_errors, self.zh_reference),
            "wer": self._rate(self.en_errors, self.en_reference),
            "sentences": self.sentences,
            "mixed_errors": self.mixed_errors,
            "mixed_reference": self.mixed_reference,
            "zh_reference": self.zh_reference,
            "en_reference": self.en_reference,
        }
