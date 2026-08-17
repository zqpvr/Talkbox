"""Language table.

The set is constrained to what XTTS-v2 can actually speak, since every
translation has to come back out as cloned audio. Each entry carries the three
different codes the pipeline stages want: Whisper's ISO-639-1, NLLB's
FLORES-200 tag, and XTTS's own slightly-off dialect codes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    name: str
    whisper: str
    nllb: str
    xtts: str


LANGUAGES: tuple[Language, ...] = (
    Language("English", "en", "eng_Latn", "en"),
    Language("Spanish", "es", "spa_Latn", "es"),
    Language("French", "fr", "fra_Latn", "fr"),
    Language("German", "de", "deu_Latn", "de"),
    Language("Italian", "it", "ita_Latn", "it"),
    Language("Portuguese", "pt", "por_Latn", "pt"),
    Language("Polish", "pl", "pol_Latn", "pl"),
    Language("Turkish", "tr", "tur_Latn", "tr"),
    Language("Russian", "ru", "rus_Cyrl", "ru"),
    Language("Dutch", "nl", "nld_Latn", "nl"),
    Language("Czech", "cs", "ces_Latn", "cs"),
    Language("Arabic", "ar", "arb_Arab", "ar"),
    Language("Chinese", "zh", "zho_Hans", "zh-cn"),
    Language("Japanese", "ja", "jpn_Jpan", "ja"),
    Language("Hungarian", "hu", "hun_Latn", "hu"),
    Language("Korean", "ko", "kor_Hang", "ko"),
    Language("Hindi", "hi", "hin_Deva", "hi"),
)

BY_WHISPER = {lang.whisper: lang for lang in LANGUAGES}
BY_NAME = {lang.name: lang for lang in LANGUAGES}

NAMES = tuple(lang.name for lang in LANGUAGES)


def get(code: str) -> Language:
    """Look up by Whisper code, falling back to English."""
    return BY_WHISPER.get(code, BY_WHISPER["en"])
