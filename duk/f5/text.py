"""Text front-end, ported from f5-tts-mlx (lucasnewman) utils.py.

Chinese goes through jieba segmentation + pypinyin (TONE3 with tone sandhi),
everything else stays character-level — the convention the F5-TTS v1 Base
checkpoint was trained with. Kept mechanically identical to upstream.
"""

from __future__ import annotations

import jieba
import mlx.core as mx
from pypinyin import Style, lazy_pinyin

_ZH_PUNCT = "。，、；：？！《》【】—…"


def convert_char_to_pinyin(text_list: list[str], polyphone: bool = True) -> list[list[str]]:
    final_text_list = []
    custom_trans = str.maketrans(
        {"“": '"', "”": '"', "‘": "'", "’": "'", ";": ","}
    )
    for text in text_list:
        char_list: list[str] = []
        text = text.translate(custom_trans)
        for seg in jieba.cut(text):
            seg_byte_len = len(bytes(seg, "UTF-8"))
            if seg_byte_len == len(seg):  # pure alphabets and symbols
                if char_list and seg_byte_len > 1 and char_list[-1] not in " :'\"":
                    char_list.append(" ")
                char_list.extend(seg)
            elif polyphone and seg_byte_len == 3 * len(seg):  # pure chinese
                pinyin = lazy_pinyin(seg, style=Style.TONE3, tone_sandhi=True)
                for c in pinyin:
                    if c not in _ZH_PUNCT:
                        char_list.append(" ")
                    char_list.append(c)
            else:  # mixed
                for c in seg:
                    if ord(c) < 256:
                        char_list.extend(c)
                    elif c not in _ZH_PUNCT:
                        char_list.append(" ")
                        char_list.extend(
                            lazy_pinyin(c, style=Style.TONE3, tone_sandhi=True)
                        )
                    else:
                        char_list.append(c)
        final_text_list.append(char_list)
    return final_text_list


def tokens_to_ids(tokens: list[str], vocab: dict[str, int]) -> mx.array:
    """One token sequence -> (1, nt) int array; unknown tokens map to 0."""
    return mx.array([[vocab.get(tok, 0) for tok in tokens]])
