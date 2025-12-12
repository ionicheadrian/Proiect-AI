import torch
import torch.nn as nn
import sentencepiece as spm
import numpy as np
from difflib import SequenceMatcher
import re

# ----------------------------------
# DEVICE / CONSTANTS
# ----------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128

# ----------------------------------
# MODEL
# ----------------------------------
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, max_len=128, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

    def create_padding_mask(self, seq):
        return (seq == 0)

    def forward(self, src, tgt):
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        src_padding_mask = self.create_padding_mask(src)
        tgt_padding_mask = self.create_padding_mask(tgt)

        src_pos = torch.arange(src.size(1), device=src.device).unsqueeze(0)
        tgt_pos = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0)

        src_emb = self.dropout(
            self.embedding(src) * np.sqrt(self.d_model) + self.pos_encoder(src_pos)
        )
        tgt_emb = self.dropout(
            self.embedding(tgt) * np.sqrt(self.d_model) + self.pos_encoder(tgt_pos)
        )

        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        output = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        return self.output_layer(output)

# ----------------------------------
# GENERATION HELPERS
# ----------------------------------
def prepare_sequence(ids, max_len):
    ids = ids[:max_len-2]
    ids = [2] + ids + [3]
    return ids + [0] * (max_len - len(ids))


def generate(model, sp, src_text, max_len=128, temperature=0.7, top_k=50, top_p=0.9):
    model.eval()
    src_ids = prepare_sequence(sp.EncodeAsIds(src_text), MAX_LEN)
    src = torch.tensor([src_ids]).to(DEVICE)
    tgt_ids = torch.tensor([[2]], device=DEVICE)

    with torch.no_grad():
        for _ in range(max_len):
            output = model(src, tgt_ids)
            logits = output[:, -1, :] / temperature

            if top_k > 0:
                indices = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            if next_token.item() == 3:
                break

            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)

    return sp.DecodeIds(tgt_ids[0].tolist()).replace("<s>", "").replace("</s>", "").strip()

# ----------------------------------
# SIMILARITY
# ----------------------------------
def calculate_similarity(user_answer, correct_answer):
    def normalize(text):
        return re.sub(r"[^\w\s]", "", text.lower().strip())

    u = normalize(user_answer)
    c = normalize(correct_answer)

    if u == c:
        return 100

    seq = SequenceMatcher(None, u, c).ratio() * 100
    uw, cw = set(u.split()), set(c.split())
    word = (len(uw & cw) / len(uw | cw)) * 100 if cw else 0

    return round(0.6 * seq + 0.4 * word)

# ----------------------------------
# LOAD MODELS (ONCE)
# ----------------------------------
print("Loading models...")

sp_answer = spm.SentencePieceProcessor(model_file="../modelCA/tokenizer.model")
answer_model = TransformerSeq2Seq(sp_answer.GetPieceSize()).to(DEVICE)
answer_model.load_state_dict(torch.load("../modelCA/best_model.pt", map_location=DEVICE)["model_state_dict"])
answer_model.eval()

sp_question = spm.SentencePieceProcessor(model_file="../modelAQ/tokenizer.model")
question_model = TransformerSeq2Seq(sp_question.GetPieceSize()).to(DEVICE)
question_model.load_state_dict(torch.load("../modelAQ/best_model.pt", map_location=DEVICE)["model_state_dict"])
question_model.eval()

print("✅ Models loaded")

# ----------------------------------
# PUBLIC API FOR UI
# ----------------------------------
def generate_question(category, temperature):
    answer = generate(answer_model, sp_answer, category, temperature=temperature)
    question = generate(question_model, sp_question, answer, temperature=temperature)
    return question, answer
