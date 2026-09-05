"""
translator.py

Language detection + translation, kept on PRETRAINED models (see the
project README for why: there's no parallel email corpus to fine-tune on).

Uses:
  - langdetect for detecting the source language
  - facebook/nllb-200-distilled-600M for translation (broad language coverage,
    good for Indian languages too: Hindi, Marathi, etc.)

NOTE: loads the model/tokenizer directly with AutoModelForSeq2SeqLM instead
of transformers.pipeline("translation", ...) — newer transformers versions
(v5+) restructured the pipeline task registry and dropped/renamed some task
strings, so this avoids relying on that lookup entirely.
"""

import torch
from langdetect import detect, LangDetectException
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# langdetect ISO 639-1 code -> NLLB FLORES-200 code
# extend this dict if you need more languages
LANGDETECT_TO_NLLB = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "ar": "arb_Arab",
    "ru": "rus_Cyrl",
    "pt": "por_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "gu": "guj_Gujr",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "bn": "ben_Beng",
}


class Translator:
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-600M"):
        self._model_name = model_name
        self._tokenizer = None
        self._model = None  # lazy-loaded, translation model is large
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self):
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name).to(self._device)
            self._model.eval()

    def detect_language(self, text: str) -> str:
        """Returns a langdetect code, e.g. 'en', 'hi'. Defaults to 'en' on failure."""
        try:
            return detect(text)
        except LangDetectException:
            return "en"

    def translate(self, text: str, target_lang: str = "en") -> str:
        """
        Translates `text` into `target_lang` (langdetect-style code, e.g. 'en').
        Returns the original text unchanged if source == target or the
        language isn't in our NLLB mapping.
        """
        src_lang = self.detect_language(text)
        if src_lang == target_lang:
            return text

        src_code = LANGDETECT_TO_NLLB.get(src_lang)
        tgt_code = LANGDETECT_TO_NLLB.get(target_lang)
        if src_code is None or tgt_code is None:
            # unsupported language pair, return as-is rather than crash
            return text

        self._load()
        self._tokenizer.src_lang = src_code
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self._device)

        forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(tgt_code)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
            )

        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    t = Translator()
    sample = "Namaste, kripya is email ka jawab jaldi de."
    lang = t.detect_language(sample)
    print(f"Detected language: {lang}")
    print("Translated:", t.translate(sample, target_lang="en"))