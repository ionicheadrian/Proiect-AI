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
# IMPROVED SIMILARITY EVALUATION
# ----------------------------------
def calculate_similarity(user_answer, correct_answer):
    """
    Calculate similarity between user answer and correct answer using multiple methods.
    Returns a score from 0-100.
    """
    def normalize(text):
        """Normalize text for comparison"""
        text = text.lower().strip()
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove punctuation but keep apostrophes for contractions
        text = re.sub(r'[^\w\s\']', '', text)
        return text
    
    def tokenize(text):
        """Split text into words"""
        return text.split()
    
    user_norm = normalize(user_answer)
    correct_norm = normalize(correct_answer)
    
    # Empty answer check
    if not user_norm:
        return 0
    
    # Exact match after normalization
    if user_norm == correct_norm:
        return 100
    
    # 1. Character-level similarity (using SequenceMatcher)
    seq_matcher = SequenceMatcher(None, user_norm, correct_norm)
    char_similarity = seq_matcher.ratio() * 100
    
    # 2. Word-level analysis
    user_words = set(tokenize(user_norm))
    correct_words = set(tokenize(correct_norm))
    
    if not correct_words:
        return 0
    
    # Jaccard similarity (intersection over union)
    intersection = user_words & correct_words
    union = user_words | correct_words
    jaccard_score = (len(intersection) / len(union)) * 100 if union else 0
    
    # Precision: how many user words are correct
    precision = (len(intersection) / len(user_words)) * 100 if user_words else 0
    
    # Recall: how many correct words did user include
    recall = (len(intersection) / len(correct_words)) * 100 if correct_words else 0
    
    # F1 score (harmonic mean of precision and recall)
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    # 3. Longest Common Subsequence (LCS) based similarity
    def lcs_length(s1, s2):
        """Calculate longest common subsequence length"""
        words1 = tokenize(s1)
        words2 = tokenize(s2)
        m, n = len(words1), len(words2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words1[i-1] == words2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    lcs_len = lcs_length(user_norm, correct_norm)
    correct_word_count = len(tokenize(correct_norm))
    lcs_score = (lcs_len / correct_word_count) * 100 if correct_word_count > 0 else 0
    
    # 4. Position-aware similarity (gives bonus for words in correct order)
    user_words_list = tokenize(user_norm)
    correct_words_list = tokenize(correct_norm)
    
    position_matches = 0
    for i, word in enumerate(user_words_list):
        if i < len(correct_words_list) and word == correct_words_list[i]:
            position_matches += 1
    
    position_score = (position_matches / max(len(user_words_list), len(correct_words_list))) * 100
    
    # 5. Length penalty (significantly different lengths get penalized)
    len_ratio = min(len(user_norm), len(correct_norm)) / max(len(user_norm), len(correct_norm))
    length_penalty = len_ratio if len_ratio > 0.3 else 0.3  # Don't penalize too harshly
    
    # Weighted combination of all metrics
    # Character similarity: 25% (good for typos)
    # F1 score: 30% (balanced precision/recall)
    # Jaccard: 20% (word overlap)
    # LCS: 15% (sequential matching)
    # Position: 10% (word order)
    
    word_based_score = (f1_score * 0.3 + jaccard_score * 0.2 + lcs_score * 0.15 + position_score * 0.1) / 0.75
    weighted_word_score = word_based_score * length_penalty
    
    final_score = (char_similarity * 0.25) + (weighted_word_score * 0.75)
    
    # Ensure score is between 0 and 100
    final_score = max(0, min(100, final_score))
    
    return round(final_score)

# ----------------------------------
# LOAD MODELS (ONCE)
# ----------------------------------
import os

# Get the directory where backend.py is located
backend_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
project_root = os.path.dirname(backend_dir)

print("Loading models...")

# Build paths relative to project root
model_ca_path = os.path.join(project_root, "modelCA", "tokenizer.model")
model_ca_checkpoint = os.path.join(project_root, "modelCA", "best_model.pt")
model_aq_path = os.path.join(project_root, "modelAQ", "tokenizer.model")
model_aq_checkpoint = os.path.join(project_root, "modelAQ", "best_model.pt")

sp_answer = spm.SentencePieceProcessor(model_file=model_ca_path)
answer_model = TransformerSeq2Seq(sp_answer.GetPieceSize()).to(DEVICE)
answer_model.load_state_dict(torch.load(model_ca_checkpoint, map_location=DEVICE)["model_state_dict"])
answer_model.eval()

sp_question = spm.SentencePieceProcessor(model_file=model_aq_path)
question_model = TransformerSeq2Seq(sp_question.GetPieceSize()).to(DEVICE)
question_model.load_state_dict(torch.load(model_aq_checkpoint, map_location=DEVICE)["model_state_dict"])
question_model.eval()

print("✅ Models loaded")

# ----------------------------------
# PUBLIC API FOR UI
# ----------------------------------
def generate_question(category, temperature):
    """Generate a question and answer pair for the given category"""
    answer = generate(answer_model, sp_answer, category, temperature=temperature)
    question = generate(question_model, sp_question, answer, temperature=temperature)
    return question, answer