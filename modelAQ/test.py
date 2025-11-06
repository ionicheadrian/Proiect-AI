import json
import torch
import sentencepiece as spm
import torch.nn as nn
import numpy as np
from collections import defaultdict

# -------------------------------
# Load model and tokenizer
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128

# Load tokenizer
sp = spm.SentencePieceProcessor(model_file="tokenizer.model")
vocab_size = sp.GetPieceSize()
print(f"Loaded tokenizer with vocab size: {vocab_size}")

# Load your model architecture (must match training)
class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, 
                 dim_feedforward=256, max_len=128, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(max_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, 
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
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

# Load model
model = TransformerSeq2Seq(vocab_size=vocab_size).to(DEVICE)
checkpoint = torch.load("best_model.pt", map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ Model loaded (best val loss: {checkpoint['val_loss']:.4f})")

# -------------------------------
# Generation function
# -------------------------------
def prepare_sequence(ids, max_len):
    ids = ids[:max_len-2]
    ids = [2] + ids + [3]
    ids += [0]*(max_len - len(ids))
    return ids

def generate(model, sp, src_text, max_len=128, temperature=0.7, top_k=50, top_p=0.9):
    """Generate text with multiple sampling strategies"""
    model.eval()
    src_ids = prepare_sequence(sp.EncodeAsIds(src_text), MAX_LEN)
    src = torch.tensor([src_ids]).to(DEVICE)
    tgt_ids = torch.tensor([[2]], device=DEVICE)  # BOS
    
    with torch.no_grad():
        for _ in range(max_len):
            output = model(src, tgt_ids)
            logits = output[:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = float('-inf')
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            if next_token.item() == 3:  # EOS
                break
            
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
    
    return sp.DecodeIds(tgt_ids[0].tolist()).replace('<s>', '').replace('</s>', '').strip()

# -------------------------------
# TEST 1: Generate from different problem types
# -------------------------------
print("\n" + "="*70)
print("TEST 1: GENERATION FROM DIFFERENT PROBLEM TYPES")
print("="*70)

problem_types = [
    "csp_backtracking",
    "linear_programming", 
    "graph_coloring",
    "knapsack",
    "scheduling"
]

for ptype in problem_types:
    input_text = f"[PROBLEM_TYPE:{ptype}]"
    print(f"\n📝 Problem Type: {ptype}")
    print("-" * 70)
    for i in range(3):
        generated = generate(model, sp, input_text, temperature=0.7)
        print(f"  {i+1}. {generated}")

# -------------------------------
# TEST 2: Temperature comparison
# -------------------------------
print("\n" + "="*70)
print("TEST 2: TEMPERATURE EFFECTS (same input, different temperatures)")
print("="*70)

test_input = "[PROBLEM_TYPE:csp_backtracking]"
temperatures = [0.3, 0.7, 1.0, 1.5]

for temp in temperatures:
    print(f"\n🌡️  Temperature = {temp}")
    print("-" * 70)
    for i in range(2):
        generated = generate(model, sp, test_input, temperature=temp)
        print(f"  {i+1}. {generated}")

# -------------------------------
# TEST 3: Evaluate on test set (if available)
# -------------------------------
print("\n" + "="*70)
print("TEST 3: EVALUATION ON TEST SET")
print("="*70)

try:
    with open("qa_question_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # Use test split (last 10%)
    test_data = dataset[int(0.9 * len(dataset)):]
    
    if len(test_data) == 0:
        print("⚠️  No test data available")
    else:
        print(f"Testing on {len(test_data)} examples\n")
        
        for i, ex in enumerate(test_data[:5]):  # Show first 5
            input_text = f"[PROBLEM_TYPE:{ex['problem_type']}]"
            generated = generate(model, sp, input_text, temperature=0.7)
            
            print(f"Example {i+1}:")
            print(f"  Input: {input_text}")
            print(f"  Generated: {generated}")
            print(f"  Target:    {ex['question_text']}")
            print("-" * 70)

except FileNotFoundError:
    print("⚠️  Could not find qa_question_dataset.json")

# -------------------------------
# TEST 4: Quality metrics
# -------------------------------
print("\n" + "="*70)
print("TEST 4: GENERATION QUALITY METRICS")
print("="*70)

def analyze_generations(model, sp, problem_type, n_samples=20):
    """Analyze quality of generated text"""
    input_text = f"[PROBLEM_TYPE:{problem_type}]"
    generations = []
    
    for _ in range(n_samples):
        gen = generate(model, sp, input_text, temperature=0.8)
        generations.append(gen)
    
    # Calculate metrics
    lengths = [len(g.split()) for g in generations]
    unique_gens = len(set(generations))
    unique_ratio = unique_gens / n_samples
    
    # Check for common failure modes
    empty_count = sum(1 for g in generations if len(g.strip()) < 5)
    repetitive_count = sum(1 for g in generations if len(set(g.split())) < len(g.split()) * 0.3)
    
    print(f"\n📊 Analysis for {problem_type}:")
    print(f"  Samples: {n_samples}")
    print(f"  Unique outputs: {unique_gens} ({unique_ratio*100:.1f}%)")
    print(f"  Avg length: {np.mean(lengths):.1f} words")
    print(f"  Length range: {min(lengths)} - {max(lengths)} words")
    print(f"  Empty/too short: {empty_count} ({empty_count/n_samples*100:.1f}%)")
    print(f"  Highly repetitive: {repetitive_count} ({repetitive_count/n_samples*100:.1f}%)")
    
    # Show diversity
    print(f"\n  Sample outputs:")
    for i, gen in enumerate(generations[:3]):
        print(f"    {i+1}. {gen}")
    
    return {
        'unique_ratio': unique_ratio,
        'avg_length': np.mean(lengths),
        'empty_rate': empty_count / n_samples,
        'repetitive_rate': repetitive_count / n_samples
    }

# Test multiple problem types
for ptype in ["csp_backtracking", "linear_programming", "graph_coloring"]:
    metrics = analyze_generations(model, sp, ptype, n_samples=20)

# -------------------------------
# TEST 5: Interactive testing
# -------------------------------
print("\n" + "="*70)
print("TEST 5: INTERACTIVE MODE")
print("="*70)
print("Enter problem types to generate questions (or 'quit' to exit)")
print("Example: csp_backtracking")

while True:
    user_input = input("\n🔤 Problem type: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    
    if not user_input:
        continue
    
    input_text = f"[PROBLEM_TYPE:{user_input}]"
    
    print("\n🎲 Generating 3 variations:")
    for i in range(3):
        generated = generate(model, sp, input_text, temperature=0.8)
        print(f"  {i+1}. {generated}")

print("\n✅ Testing complete!")

# -------------------------------
# SUMMARY: What to look for
# -------------------------------
print("\n" + "="*70)
print("📋 QUALITY CHECKLIST:")
print("="*70)
print("""
✅ GOOD signs:
  - Generated text is readable and grammatical
  - Different inputs produce different outputs
  - Outputs are diverse (unique_ratio > 0.7)
  - Length is reasonable (10-30 words)
  - Questions make sense for the problem type

❌ BAD signs:
  - Same output repeated every time (unique_ratio < 0.3)
  - Gibberish or random characters
  - Very short outputs (< 5 words)
  - Highly repetitive words within single output
  - Output doesn't relate to problem type

🔧 If quality is bad:
  1. Check dataset size (need 500+ examples minimum)
  2. Try lower temperature (0.5-0.6)
  3. Check if model is just memorizing (test on new problem types)
  4. Consider using pretrained model (T5, BART) instead
""")
