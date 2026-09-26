"""
decoding_strategies.py

Hand-written greedy / beam / top-k decoding for your Seq2Seq summarizer,
replacing model.generate(num_beams=4) with explicit step-by-step loops
(NLP Module 5). No training -- pure inference on the model you already have.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput


def _next_logits(model, enc_hidden, enc_mask, dec_ids):
    out = model(
        encoder_outputs=BaseModelOutput(last_hidden_state=enc_hidden),
        attention_mask=enc_mask,
        decoder_input_ids=dec_ids,
    )
    return out.logits[:, -1, :]


@torch.no_grad()
def greedy_decode(model, input_ids, attention_mask, max_length=60, min_length=8):
    start = model.config.decoder_start_token_id or model.config.bos_token_id
    eos = model.config.eos_token_id
    enc = model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    dec = torch.tensor([[start]], device=input_ids.device)
    for step in range(max_length):
        logits = _next_logits(model, enc, attention_mask, dec)[0]
        if step < min_length:
            logits[eos] = float("-inf")
        nxt = int(torch.argmax(logits))
        dec = torch.cat([dec, torch.tensor([[nxt]], device=dec.device)], dim=1)
        if nxt == eos:
            break
    return dec[0, 1:].tolist()


@torch.no_grad()
def top_k_sample_decode(model, input_ids, attention_mask, k=40, temperature=0.8, max_length=60, seed=0):
    start = model.config.decoder_start_token_id or model.config.bos_token_id
    eos = model.config.eos_token_id
    gen = torch.Generator(device=input_ids.device).manual_seed(seed)
    enc = model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    dec = torch.tensor([[start]], device=input_ids.device)
    for _ in range(max_length):
        logits = _next_logits(model, enc, attention_mask, dec)[0] / temperature
        top_vals, top_idx = torch.topk(logits, k)
        probs = F.softmax(top_vals, dim=-1)
        nxt = int(top_idx[torch.multinomial(probs, 1, generator=gen)])
        dec = torch.cat([dec, torch.tensor([[nxt]], device=dec.device)], dim=1)
        if nxt == eos:
            break
    return dec[0, 1:].tolist()


@torch.no_grad()
def beam_search_decode(model, input_ids, attention_mask, num_beams=4, max_length=60, length_penalty=1.0):
    start = model.config.decoder_start_token_id or model.config.bos_token_id
    eos = model.config.eos_token_id
    device = input_ids.device
    enc = model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    beams = torch.tensor([[start]], device=device)
    scores = torch.zeros(1, device=device)
    finished = []

    for _ in range(max_length):
        logits = _next_logits(model, enc, attention_mask, beams)
        log_probs = F.log_softmax(logits, dim=-1)
        V = log_probs.size(-1)
        cand = (scores.unsqueeze(1) + log_probs).view(-1)
        top_scores, top_idx = torch.topk(cand, min(2 * num_beams, cand.numel()))

        new_beams, new_scores = [], []
        for score, flat in zip(top_scores.tolist(), top_idx.tolist()):
            i, tok = divmod(flat, V)
            if tok == eos:
                finished.append((score / (beams.size(1) ** length_penalty), beams[i, 1:].tolist() + [eos]))
            else:
                new_beams.append(torch.cat([beams[i], torch.tensor([tok], device=device)]))
                new_scores.append(score)
            if len(new_beams) == num_beams:
                break
        if not new_beams or len(finished) >= num_beams:
            break
        beams, scores = torch.stack(new_beams), torch.tensor(new_scores, device=device)

    if not finished:
        finished = [(s / (beams.size(1) ** length_penalty), b[1:].tolist()) for b, s in zip(beams, scores.tolist())]
    return max(finished, key=lambda x: x[0])[1]


if __name__ == "__main__":
    model_dir = "saved_models/summarizer"
    import os
    name = model_dir if os.path.isdir(model_dir) and os.listdir(model_dir) else "facebook/bart-large-cnn"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).eval()

    email = ("Hi team, following up on the Q3 budget. We need to finalize the marketing "
             "spend by Friday. Please review the attached sheet and reply with comments.")
    enc = tok(email, return_tensors="pt", truncation=True, max_length=384)

    print("greedy:", tok.decode(greedy_decode(model, enc["input_ids"], enc["attention_mask"]), skip_special_tokens=True))
    print("beam x4:", tok.decode(beam_search_decode(model, enc["input_ids"], enc["attention_mask"]), skip_special_tokens=True))
    print("top-k:", tok.decode(top_k_sample_decode(model, enc["input_ids"], enc["attention_mask"]), skip_special_tokens=True))