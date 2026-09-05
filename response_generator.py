"""
response_generator.py

Drafts a short professional reply using flan-t5, lightly templated with
a priority-based greeting.

NOTE: loads the model/tokenizer directly with AutoModelForSeq2SeqLM
instead of transformers.pipeline("text2text-generation", ...) — newer
transformers versions (v5+) restructured the pipeline task registry and
dropped/renamed some task strings, so this avoids relying on that lookup
entirely.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

GREETING_BY_PRIORITY = {
    "High": "Thanks for flagging this — replying right away.",
    "Medium": "Thanks for your email.",
    "Low": "Thanks for the note.",
}

SIGN_OFF = "Best regards,"


class ResponseGenerator:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self._model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self):
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name).to(self._device)
            self._model.eval()

    def generate(self, email_summary: str, priority: str = "Medium", sender_name: str = "") -> str:
        self._load()

        action_hint = (
            "This email requires no action from the recipient — do not ask them to "
            "review anything or request a follow-up."
            if priority == "Low"
            else "Only mention next steps or a follow-up if the summary implies an action is needed."
        )

        prompt = (
            "You are the recipient of an email, replying to the sender. "
            "The summary below describes what THEY wrote to you — do not repeat it "
            "in first person as if it were your own request. Instead, acknowledge "
            "it and respond appropriately (e.g. confirm, approve, or say you'll "
            "follow up), from your own point of view as the person replying.\n"
            "Write a short, polite professional reply, under 4 sentences. " + action_hint + "\n"
            f"Summary of their email: {email_summary}\n"
            f"Priority: {priority}\n"
            "Your reply (do not restate their request as your own):"
        )

        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_length=120,
                min_length=15,
                do_sample=False,
                num_beams=4,
                repetition_penalty=1.8,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )

        body = self._tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        greeting = GREETING_BY_PRIORITY.get(priority, GREETING_BY_PRIORITY["Medium"])
        name_line = f"Hi {sender_name}," if sender_name else "Hi,"

        return f"{name_line}\n\n{greeting} {body}\n\n{SIGN_OFF}"


if __name__ == "__main__":
    gen = ResponseGenerator()
    reply = gen.generate(
        email_summary="Client is asking to move Thursday's meeting to Friday morning.",
        priority="High",
        sender_name="Rahul",
    )
    print(reply)