"""Optional transcript-only PnC removal after word alignment and boundary scheduling."""

import re
import unicodedata


def remove_transcript_pnc(messages, *, protected_tokens, blank_token):
    """Return new messages, lowercasing lexical text and removing Unicode punctuation.

    Structural tokens are matched literally and preserved. Apostrophes are deleted
    (don't -> dont); other punctuation becomes whitespace (well-being -> well being).
    Numbers and non-punctuation symbols are not verbalized. Message/chunk boundaries,
    non-assistant messages, and alignment/reference objects are never modified.
    """
    tokens = frozenset(t for t in protected_tokens if t)
    pattern = re.compile("(" + "|".join(re.escape(t) for t in sorted(tokens, key=len, reverse=True)) + ")")
    result = []
    for message in messages:
        if message["role"] != "assistant":
            result.append(message)
            continue
        pieces = []
        for part in pattern.split(message["content"]):
            if part in tokens:
                pieces.append(part)
            else:
                normalized = "".join(
                    "" if char in "'’ʼ" else " " if unicodedata.category(char).startswith("P") else char
                    for char in part.lower()
                )
                pieces.append(re.sub(r"\s+", " ", normalized))
        content = "".join(pieces)
        result.append({**message, "content": content if content.strip() else blank_token})
    return result
