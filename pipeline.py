"""
pipeline.py

Wires together: language detection -> translation -> priority
classification -> summarization -> response generation.

This is the single entry point the Streamlit app calls.
"""

from translator import Translator
from priority_classifier import PriorityClassifier
from summarizer import Summarizer
from response_generator import ResponseGenerator

class EmailAssistantPipeline:
    def __init__(self):
        self.translator = Translator()
        self.priority_clf = PriorityClassifier()
        self.summarizer = Summarizer()
        self.responder = ResponseGenerator()

    def process(self, raw_email_text: str, sender_name: str = "") -> dict:
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

        response_en = self.responder.generate(
            email_summary=summary,
            priority=priority,
            sender_name=sender_name
        )

        # translate the drafted response back to the sender's language, if needed
        if detected_lang != "en":
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
        
