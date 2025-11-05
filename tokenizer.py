import json
import sentencepiece as spm

# -------------------------------
# Step 1: Load JSON dataset
# -------------------------------
INPUT_JSON = "qa_question_dataset.json"
OUTPUT_JSON = "qa_dataset_seq2seq.json"
CORPUS_FILE = "corpus.txt"
TOKENIZER_PREFIX = "tokenizer"
VOCAB_SIZE = 322

with open(INPUT_JSON, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# -------------------------------
# Step 2: Convert to seq2seq format
# -------------------------------
for ex in dataset:
    # Input sequence includes problem type + question
    ex["input_text"] = f"problem_type: {ex['problem_type']}; question: {ex['question_text']}"
    # Target sequence is the answer text (can be empty if generating questions)
    ex["target_text"] = ex["answer_text"]

# Save updated JSON (optional)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

# -------------------------------
# Step 3: Save corpus for tokenizer
# -------------------------------
with open(CORPUS_FILE, "w", encoding="utf-8") as f:
    for ex in dataset:
        f.write(ex["input_text"] + "\n")
        f.write(ex["target_text"] + "\n")  # include empty lines if target is empty

# -------------------------------
# Step 4: Train SentencePiece tokenizer
# -------------------------------
spm.SentencePieceTrainer.Train(
    input=CORPUS_FILE,
    model_prefix=TOKENIZER_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type='unigram',  # or 'bpe'
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

print(f"✅ Tokenizer trained and saved as {TOKENIZER_PREFIX}.model and {TOKENIZER_PREFIX}.vocab")
