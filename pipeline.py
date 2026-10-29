"""
pipeline.py

Wires together: language detection -> translation -> priority
classification -> summarization -> reply drafting.

This is the core pipeline orchestrator called by the FastAPI backend (api.py).
"""

from translator import Translator
from priority_classifier import PriorityClassifier
from summarizer import Summarizer
from template_reply import build_reply


class EmailAssistantPipeline:
    def __init__(self, use_generative_reply: bool = False):
        self.translator = Translator()
        self.priority_clf = PriorityClassifier()
        self.summarizer = Summarizer()
        self.use_generative_reply = use_generative_reply
        # Lazy reference for optional model-based response generation (flan-t5)
        self._responder = None

    @property
    def responder(self):
        """
        Lazy-load ResponseGenerator only if explicitly requested.
        Kept for research/comparison, but unused by default to avoid hallucination.
        """
        if self._responder is None:
            from response_generator import ResponseGenerator
            self._responder = ResponseGenerator()
        return self._responder

    def process(
        self,
        raw_email_text: str,
        sender_name: str = "",
        extracted_info: list = None,
        use_generative_reply: bool = None,
        translate_reply: bool = False,
    ) -> dict:
        """
        Runs the full analysis pipeline on raw email text:
        1. Detect language (and translate to English if non-English)
        2. Classify priority (DistilBERT)
        3. Summarize content (T5-small on AESLC)
        4. Draft suggested reply (grounded template by default, or flan-t5 if requested)
        5. Translate drafted reply back to sender's original language if needed
        """
        if use_generative_reply is None:
            use_generative_reply = self.use_generative_reply

        # Print the original/raw email for debugging
        print("Raw email:", repr(raw_email_text))

        detected_lang = self.translator.detect_language(raw_email_text)

        # translate to English for the models that expect it, if needed
        if detected_lang != "en":
            english_text = self.translator.translate(
                raw_email_text,
                target_lang="en"
            )
        else:
            english_text = raw_email_text

        priority = self.priority_clf.predict(english_text)

        summary = self.summarizer.summarize(english_text)

        # Draft response:
        # Note on Design / Viva Defense:
        # Free-form generation via FLAN-T5 was dropped because LLMs risk hallucinating
        # unconfirmed dates, commitments, or facts. Instead, we default to grounded,
        # rule-based templates (template_reply.py) driven by extracted key info.
        if use_generative_reply:
            response_en = self.responder.generate(
                email_summary=summary,
                priority=priority,
                sender_name=sender_name
            )
        else:
            response_en = build_reply(
                key_info=extracted_info or [],
                priority=priority,
                sender_name=sender_name
            )

        # Translate reply back to sender's language only if requested (api.py handles final translation)
        if translate_reply and detected_lang != "en":
            response_final = self.translator.translate(
                response_en,
                target_lang=detected_lang
            )
        else:
            response_final = response_en

        return {
            "detected_language": detected_lang,
            "translated_input": english_text if detected_lang != "en" else None,
            "priority": priority,
            "summary": summary,
            "suggested_response": response_final,
        }


if __name__ == "__main__":
    pipeline = EmailAssistantPipeline()

    sample_email = (
        "Hi, URGENT - we need the signed contract back by end of day today. "
        "Please review the attached PDF and send your signature ASAP."
    )

    result = pipeline.process(
        sample_email,
        sender_name="Yashika"
    )

    for key, value in result.items():
        print(f"{key}: {value}\n")
