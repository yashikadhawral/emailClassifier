"""
template_reply.py

Rule-based (non-generative) reply drafting, built entirely from fields
already extracted by key_info_extractor.py -- no language model involved,
so it never hallucinates specifics that aren't actually in the email.

Trade-off vs response_generator.py's flan-t5 output: this reads more
"templated" and won't handle nuance a model could catch, but every claim
in the output is grounded in something the extractor actually found. Use
this as the default; response_generator.py's model-based draft can still
be shown separately if you want to compare or blend the two later.
"""

PRIORITY_OPENERS = {
    "urgent": "Thanks for flagging this — I'll prioritize it.",
    "important": "Thanks for the email — I'll look into this soon.",
    "normal": "Thanks for the email.",
    "low": "Thanks for the note.",
}

PRIORITY_CLOSERS = {
    "urgent": "I'll get back to you as soon as possible.",
    "important": "I'll follow up shortly.",
    "normal": "Let me know if there's anything else you need in the meantime.",
    "low": "No action needed on my end for now.",
}


def build_reply(key_info: list, priority: str, sender_name: str = "") -> str:
    """
    key_info: the list of {"k": ..., "v": ...} dicts from key_info_extractor.
    priority: one of "urgent" | "important" | "normal" | "low" (frontend's scale).
    """
    kv = {field["k"]: field["v"] for field in key_info}

    deadline = kv.get("Deadline")
    doc = kv.get("Referenced doc")

    greeting = f"Hi {sender_name}," if sender_name and sender_name != "Unknown sender" else "Hi,"
    opener = PRIORITY_OPENERS.get(priority, PRIORITY_OPENERS["normal"])

    middle_parts = []
    if doc:
        middle_parts.append(f"I'll review {doc}")
    if deadline:
        if middle_parts:
            middle_parts.append(f"and get back to you by {deadline}")
        else:
            middle_parts.append(f"I've noted the timeline ({deadline}) and will act accordingly")

    if middle_parts:
        middle = " ".join(middle_parts) + "."
    else:
        middle = "I'll look into this and get back to you."

    closer = PRIORITY_CLOSERS.get(priority, PRIORITY_CLOSERS["normal"])

    return f"{greeting}\n\n{opener} {middle} {closer}\n\nBest regards,"


if __name__ == "__main__":
    sample_key_info = [
        {"k": "Deadline", "v": "Thu, 11:00 AM"},
        {"k": "Referenced doc", "v": "Q3_eval_slides_v4.pptx"},
    ]
    print(build_reply(sample_key_info, priority="urgent", sender_name="Priya"))