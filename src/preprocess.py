"""Sarcasm-aware text normalisation.

Design choices (see docs/overview): keep emojis and punctuation, keep stop words,
collapse stretched characters, leave casing information to the model tokenizer.
"""
import re
import unicodedata

import emoji

_STRETCH = re.compile(r"(.)\1{2,}")          # waaaah -> waah
_PUNCT_RUN = re.compile(r"([!?.])\1{3,}")     # !!!!!! -> !!!
_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = _URL.sub(" ", text)
    text = _MENTION.sub("@user", text)
    # real emoji characters -> words (the dataset already stores emojis as words)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = _STRETCH.sub(r"\1\1", text)
    text = _PUNCT_RUN.sub(r"\1\1\1", text)
    return _SPACES.sub(" ", text).strip()


def dedup_key(text: str) -> str:
    return _SPACES.sub(" ", unicodedata.normalize("NFKC", str(text)).lower()).strip()
