import json
import random
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# -------------------------------
# CONFIG
# -------------------------------
INPUT_JSON = "ai_dataset.json"
CORPUS_FILE = "corpus.txt"
TOKENIZER_PREFIX = "tokenizer"
VOCAB_SIZE = 300  # Will be auto-adjusted based on corpus
MAX_LEN = 128
BATCH_SIZE = 16
NUM_EPOCHS = 30  # More epochs with early stopping
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4  # Lower learning rate
GRADIENT_CLIP = 1.0
MIN_VOCAB_SIZE = 100  # Minimum reasonable vocab size

# -------------------------------
# STEP 1: Load and preprocess dataset
# -------------------------------
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Less aggressive preprocessing - keep more semantic information
def preprocess_example(ex):
    # Keep problem structure but maintain readability
    ex["input_text"] = f"[ANSWER:{ex['answer_text']}]"
    ex["target_text"] = ex["question_text"]
    return ex

dataset = [preprocess_example(ex) for ex in dataset]

# Data augmentation if dataset is small
if len(dataset) < 500:
    print(f"⚠️ Warning: Small dataset ({len(dataset)} examples). Consider collecting more data.")

# Shuffle and split
random.shuffle(dataset)
split1 = int(0.8 * len(dataset))
split2 = int(0.9 * len(dataset))
train_data = dataset[:split1]
val_data = dataset[split1:split2]
test_data = dataset[split2:]

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# -------------------------------
# STEP 2: Prepare corpus.txt and analyze it
# -------------------------------
with open(CORPUS_FILE, "w", encoding="utf-8") as f:
    for ex in dataset:
        f.write(ex["input_text"] + "\n")
        f.write(ex["target_text"] + "\n")

# Analyze corpus
print("\n" + "="*50)
print("CORPUS ANALYSIS:")
print("="*50)
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    corpus_text = f.read()
    
corpus_chars = set(corpus_text)
corpus_words = set(corpus_text.split())

print(f"Total text length: {len(corpus_text)} characters")
print(f"Unique characters: {len(corpus_chars)}")
print(f"Unique words: {len(corpus_words)}")
print(f"Sample text: {corpus_text[:200]}...")

# Calculate appropriate vocab size
estimated_vocab = len(corpus_chars) + len(corpus_words) // 2
suggested_vocab = min(max(estimated_vocab, MIN_VOCAB_SIZE), 5000)
print(f"\n⚠️ Estimated max vocab size: {estimated_vocab}")
print(f"💡 Suggested vocab size: {suggested_vocab}")

if len(dataset) < 100:
    print(f"\n⚠️ CRITICAL WARNING: Dataset has only {len(dataset)} examples!")
    print("   For good results, you need at least 500-1000 examples.")
    print("   Current results will likely be poor.")

# Use conservative vocab size
VOCAB_SIZE = min(suggested_vocab, 1000)

# -------------------------------
# STEP 3: Train SentencePiece tokenizer with auto-sizing
# -------------------------------
print(f"\n{'='*50}")
print(f"Training tokenizer with vocab_size={VOCAB_SIZE}...")
print(f"{'='*50}")

try:
    spm.SentencePieceTrainer.Train(
        input=CORPUS_FILE,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type='unigram',
        character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        normalization_rule_name='identity',
        max_sentence_length=4096
    )
    print(f"✅ Tokenizer trained successfully")
except Exception as e:
    print(f"❌ Tokenizer training failed with vocab_size={VOCAB_SIZE}")
    print(f"Error: {e}")
    print("\nTrying with smaller vocab_size...")
    # Try progressively smaller sizes
    for try_vocab in [500, 300, 200, 150, 100]:
        try:
            print(f"  Attempting vocab_size={try_vocab}...")
            spm.SentencePieceTrainer.Train(
                input=CORPUS_FILE,
                model_prefix=TOKENIZER_PREFIX,
                vocab_size=try_vocab,
                model_type='unigram',
                character_coverage=1.0,
                pad_id=0, unk_id=1, bos_id=2, eos_id=3,
                normalization_rule_name='identity',
                max_sentence_length=4096
            )
            VOCAB_SIZE = try_vocab
            print(f"✅ Success with vocab_size={VOCAB_SIZE}")
            break
        except:
            continue
    else:
        raise RuntimeError("Could not train tokenizer even with small vocab. Dataset may be too small.")

sp = spm.SentencePieceProcessor(model_file=f"{TOKENIZER_PREFIX}.model")
print(f"Final vocab size: {sp.GetPieceSize()}")

# Test tokenization
test_text = train_data[0]["target_text"] if train_data else "test"
tokens = sp.EncodeAsPieces(test_text)
print(f"\nTokenization example:")
print(f"Text: {test_text[:100]}...")
print(f"Tokens: {tokens[:20]}...")
print(f"Token count: {len(tokens)}")

# -------------------------------
# STEP 4: Prepare Dataset and DataLoader
# -------------------------------
def prepare_sequence(ids, max_len):
    ids = ids[:max_len-2]
    ids = [2] + ids + [3]  # BOS + sequence + EOS
    ids += [0]*(max_len - len(ids))
    return ids

class QADataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        src_ids = prepare_sequence(sp.EncodeAsIds(ex["input_text"]), MAX_LEN)
        tgt_ids = prepare_sequence(sp.EncodeAsIds(ex["target_text"]), MAX_LEN)
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

train_loader = DataLoader(QADataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(QADataset(val_data), batch_size=BATCH_SIZE)

# -------------------------------
# STEP 5: Improved Transformer with proper masking
# -------------------------------
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
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask to prevent looking at future tokens"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask
    
    def create_padding_mask(self, seq):
        """Create mask for padding tokens"""
        return (seq == 0)
    
    def forward(self, src, tgt):
        # Create masks
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        src_padding_mask = self.create_padding_mask(src)
        tgt_padding_mask = self.create_padding_mask(tgt)
        
        # Position embeddings
        src_pos = torch.arange(src.size(1), device=src.device).unsqueeze(0)
        tgt_pos = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0)
        
        # Embeddings with scaling
        src_emb = self.dropout(
            self.embedding(src) * np.sqrt(self.d_model) + self.pos_encoder(src_pos)
        )
        tgt_emb = self.dropout(
            self.embedding(tgt) * np.sqrt(self.d_model) + self.pos_encoder(tgt_pos)
        )
        
        # Encode
        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        
        # Decode with proper masking
        output = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        return self.output_layer(output)

model = TransformerSeq2Seq(vocab_size=sp.GetPieceSize()).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# -------------------------------
# STEP 6: Training loop with validation
# -------------------------------
best_val_loss = float('inf')
patience_counter = 0
patience_limit = 5

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    total_loss = 0
    for src, tgt in train_loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_input = tgt[:, :-1]
        tgt_labels = tgt[:, 1:]
        
        optimizer.zero_grad()
        output = model(src, tgt_input)
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_labels.reshape(-1))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        
        optimizer.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_input = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_labels.reshape(-1))
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Learning rate scheduling
    scheduler.step(avg_val_loss)
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
        }, "best_model.pt")
        print("✅ Saved best model")
    else:
        patience_counter += 1
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Load best model
checkpoint = torch.load("best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
print("✅ Loaded best model for inference")

# -------------------------------
# STEP 7: Improved inference with sampling
# -------------------------------
def generate(model, sp, src_text, max_len=128, temperature=0.8, top_k=50, top_p=0.9):
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
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = float('-inf')
            
            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            if next_token.item() == 3:  # EOS
                break
            
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
    
    return sp.DecodeIds(tgt_ids[0].tolist()).replace('<s>', '').replace('</s>', '').strip()

# Test generation
print("\n" + "="*50)
print("GENERATION EXAMPLES:")
print("="*50)

#for problem_type in ["csp_backtracking", "linear_programming", "graph_coloring"]:
#    example_input = f"[PROBLEM_TYPE:{problem_type}]"
#    print(f"\nInput: {example_input}")
#    for i in range(3):
#        print(f"  {i+1}. {generate(model, sp, example_input)}")

# Lista de tipuri de răspunsuri / output-uri pe care vrei să le testezi
answer_prompts = [
    "Backtracking with Forward Checking (FC) and Minimum Remaining Values (MRV)",
    "Local Search (Hill-Climbing with random restarts)",
    "DSATUR heuristic (Degree of Saturation)",
    "Warnsdorff's heuristic rule",
    #"Recursive optimal 3-peg algorithm",
    #"Frame–Stewart heuristic",
    #"Pure Nash equilibrium",
    #"No pure Nash equilibrium",
    #"Root value (MinMax + Alpha-Beta)",
    #"Complete assignment for CSP"
]

# Testare simplă a NN pentru a vedea ce întrebări generează
for answer_text in answer_prompts:
    example_input = f"[ANSWER:{answer_text}]"
    print(f"\nInput: {example_input}")
    for i in range(3):
        # generate(...) este funcția ta de inferență pe model
        print(f"  {i+1}. {generate(model, sp, example_input)}")
