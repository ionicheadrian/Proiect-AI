import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import torch
import torch.nn as nn
import sentencepiece as spm
import numpy as np
from difflib import SequenceMatcher
import re

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 128

# Model Architecture
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

# Prepare sequence helper
def prepare_sequence(ids, max_len):
    ids = ids[:max_len-2]
    ids = [2] + ids + [3]
    ids += [0]*(max_len - len(ids))
    return ids

# Assuming you have these from your model setup
# from your_model_module import prepare_sequence, MAX_LEN, DEVICE

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


class QuizApplication:
    def __init__(self, root, answer_model, question_model, sp_answer, sp_question):
        self.root = root
        self.root.title("AI Quiz Generator")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f4f8')
        
        # Models
        self.answer_model = answer_model
        self.question_model = question_model
        self.sp_answer = sp_answer
        self.sp_question = sp_question
        
        # State
        self.current_category = None
        self.current_answer = None
        self.current_question = None
        
        # Categories based on your dataset
#        self.categories = [
#            "minmax_alphabeta",
#            "search_algorithms",
#            "constraint_satisfaction",
#            "logic_inference",
#            "machine_learning",
#            "neural_networks",
#            "optimization",
#            "graph_theory",
#            "probability",
#            "reinforcement_learning"
#        ]
        self.categories = [
            "n-queens",
            "generalised_hanoi",
            "graph_coloring",
            "knights_tour",
            "csp_backtracking",
            "minmax_alphabeta",
            "game_theory"
        ]

        
        self.setup_ui()
    
    def setup_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f4f8')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="🎓 AI Quiz Generator",
            font=('Helvetica', 24, 'bold'),
            bg='#f0f4f8',
            fg='#2c3e50'
        )
        title_label.pack(pady=(0, 20))
        
        # Category Selection Frame
        self.category_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, borderwidth=2)
        self.category_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        cat_label = tk.Label(
            self.category_frame,
            text="Select Category:",
            font=('Helvetica', 14, 'bold'),
            bg='white'
        )
        cat_label.pack(pady=(20, 10))
        
        self.category_var = tk.StringVar()
        self.category_combo = ttk.Combobox(
            self.category_frame,
            textvariable=self.category_var,
            values=self.categories,
            state='readonly',
            font=('Helvetica', 12),
            width=30
        )
        self.category_combo.pack(pady=10)
        
        generate_btn = tk.Button(
            self.category_frame,
            text="Generate Question",
            command=self.generate_question,
            bg='#3498db',
            fg='white',
            font=('Helvetica', 12, 'bold'),
            padx=30,
            pady=10,
            cursor='hand2',
            relief=tk.RAISED
        )
        generate_btn.pack(pady=20)
        
        self.status_label = tk.Label(
            self.category_frame,
            text="",
            font=('Helvetica', 10, 'italic'),
            bg='white',
            fg='#7f8c8d'
        )
        self.status_label.pack(pady=10)
        
        # Question Frame (initially hidden)
        self.question_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, borderwidth=2)
        
        q_title = tk.Label(
            self.question_frame,
            text="Question:",
            font=('Helvetica', 14, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        q_title.pack(pady=(20, 10), anchor='w', padx=20)
        
        self.question_text = scrolledtext.ScrolledText(
            self.question_frame,
            wrap=tk.WORD,
            width=70,
            height=5,
            font=('Helvetica', 11),
            bg='#e8f4f8',
            relief=tk.FLAT,
            state='disabled'
        )
        self.question_text.pack(pady=10, padx=20)
        
        answer_label = tk.Label(
            self.question_frame,
            text="Your Answer:",
            font=('Helvetica', 14, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        answer_label.pack(pady=(20, 10), anchor='w', padx=20)
        
        self.user_answer_text = scrolledtext.ScrolledText(
            self.question_frame,
            wrap=tk.WORD,
            width=70,
            height=5,
            font=('Helvetica', 11)
        )
        self.user_answer_text.pack(pady=10, padx=20)
        
        submit_btn = tk.Button(
            self.question_frame,
            text="Submit Answer",
            command=self.check_answer,
            bg='#27ae60',
            fg='white',
            font=('Helvetica', 12, 'bold'),
            padx=30,
            pady=10,
            cursor='hand2',
            relief=tk.RAISED
        )
        submit_btn.pack(pady=20)
        
        # Result Frame (initially hidden)
        self.result_frame = tk.Frame(main_frame, bg='white', relief=tk.RAISED, borderwidth=2)
        
        result_title = tk.Label(
            self.result_frame,
            text="Results",
            font=('Helvetica', 18, 'bold'),
            bg='white',
            fg='#2c3e50'
        )
        result_title.pack(pady=(20, 10))
        
        self.score_label = tk.Label(
            self.result_frame,
            text="",
            font=('Helvetica', 36, 'bold'),
            bg='white'
        )
        self.score_label.pack(pady=20)
        
        self.feedback_label = tk.Label(
            self.result_frame,
            text="",
            font=('Helvetica', 12),
            bg='white'
        )
        self.feedback_label.pack(pady=10)
        
        # Correct Answer Display
        correct_title = tk.Label(
            self.result_frame,
            text="Correct Answer:",
            font=('Helvetica', 12, 'bold'),
            bg='white',
            fg='#27ae60'
        )
        correct_title.pack(pady=(20, 5), anchor='w', padx=20)
        
        self.correct_answer_text = scrolledtext.ScrolledText(
            self.result_frame,
            wrap=tk.WORD,
            width=70,
            height=4,
            font=('Helvetica', 11),
            bg='#d5f4e6',
            relief=tk.FLAT,
            state='disabled'
        )
        self.correct_answer_text.pack(pady=5, padx=20)
        
        # User Answer Display
        user_title = tk.Label(
            self.result_frame,
            text="Your Answer:",
            font=('Helvetica', 12, 'bold'),
            bg='white',
            fg='#3498db'
        )
        user_title.pack(pady=(15, 5), anchor='w', padx=20)
        
        self.user_answer_display = scrolledtext.ScrolledText(
            self.result_frame,
            wrap=tk.WORD,
            width=70,
            height=4,
            font=('Helvetica', 11),
            bg='#e3f2fd',
            relief=tk.FLAT,
            state='disabled'
        )
        self.user_answer_display.pack(pady=5, padx=20)
        
        try_again_btn = tk.Button(
            self.result_frame,
            text="Try Another Question",
            command=self.reset_quiz,
            bg='#9b59b6',
            fg='white',
            font=('Helvetica', 12, 'bold'),
            padx=30,
            pady=10,
            cursor='hand2',
            relief=tk.RAISED
        )
        try_again_btn.pack(pady=20)
    
    def generate_question(self):
        category = self.category_var.get()
        if not category:
            messagebox.showwarning("Warning", "Please select a category!")
            return
        
        self.status_label.config(text="Generating answer...")
        self.root.update()
        
        try:
            # Step 1: Generate answer from category
            self.current_answer = generate(
                self.answer_model,
                self.sp_answer,
                category,
                max_len=128,
                temperature=0.7
            )
            
            self.status_label.config(text="Generating question...")
            self.root.update()
            
            # Step 2: Generate question from answer
            self.current_question = generate(
                self.question_model,
                self.sp_question,
                self.current_answer,
                max_len=128,
                temperature=0.7
            )
            
            self.current_category = category
            
            # Display question
            self.question_text.config(state='normal')
            self.question_text.delete(1.0, tk.END)
            self.question_text.insert(1.0, self.current_question)
            self.question_text.config(state='disabled')
            
            # Switch to question frame
            self.category_frame.pack_forget()
            self.question_frame.pack(fill=tk.BOTH, expand=True, pady=10)
            
            self.status_label.config(text="")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate question: {str(e)}")
            self.status_label.config(text="")
    
    def calculate_similarity(self, user_answer, correct_answer):
        """Calculate similarity between user answer and correct answer"""
        # Normalize text
        def normalize(text):
            text = text.lower().strip()
            text = re.sub(r'[^\w\s]', '', text)
            return text
        
        user_norm = normalize(user_answer)
        correct_norm = normalize(correct_answer)
        
        # Exact match
        if user_norm == correct_norm:
            return 100
        
        # Sequence matcher for overall similarity
        seq_match = SequenceMatcher(None, user_norm, correct_norm)
        seq_score = seq_match.ratio() * 100
        
        # Word overlap
        user_words = set(user_norm.split())
        correct_words = set(correct_norm.split())
        
        if not correct_words:
            return 0
        
        intersection = user_words & correct_words
        union = user_words | correct_words
        
        word_score = (len(intersection) / len(union)) * 100 if union else 0
        
        # Weighted combination
        final_score = (seq_score * 0.6) + (word_score * 0.4)
        
        return round(final_score)
    
    def check_answer(self):
        user_answer = self.user_answer_text.get(1.0, tk.END).strip()
        
        if not user_answer:
            messagebox.showwarning("Warning", "Please enter your answer!")
            return
        
        # Calculate similarity score
        score = self.calculate_similarity(user_answer, self.current_answer)
        
        # Update score display
        self.score_label.config(text=f"{score}%")
        
        # Color code the score
        if score >= 80:
            self.score_label.config(fg='#27ae60')
            feedback = "Excellent! 🎉"
        elif score >= 60:
            self.score_label.config(fg='#f39c12')
            feedback = "Good effort! 👍"
        else:
            self.score_label.config(fg='#e74c3c')
            feedback = "Keep practicing! 💪"
        
        self.feedback_label.config(text=feedback)
        
        # Display correct answer
        self.correct_answer_text.config(state='normal')
        self.correct_answer_text.delete(1.0, tk.END)
        self.correct_answer_text.insert(1.0, self.current_answer)
        self.correct_answer_text.config(state='disabled')
        
        # Display user answer
        self.user_answer_display.config(state='normal')
        self.user_answer_display.delete(1.0, tk.END)
        self.user_answer_display.insert(1.0, user_answer)
        self.user_answer_display.config(state='disabled')
        
        # Switch to result frame
        self.question_frame.pack_forget()
        self.result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    def reset_quiz(self):
        # Clear all fields
        self.category_var.set('')
        self.user_answer_text.delete(1.0, tk.END)
        self.current_category = None
        self.current_answer = None
        self.current_question = None
        
        # Switch back to category frame
        self.result_frame.pack_forget()
        self.category_frame.pack(fill=tk.BOTH, expand=True, pady=10)


def main():
    print("Loading models...")
    
    # Load Answer Model (Category -> Answer)
    sp_answer = spm.SentencePieceProcessor(model_file="modelCA/tokenizer.model")
    vocab_size_answer = sp_answer.GetPieceSize()
    print(f"Loaded answer tokenizer with vocab size: {vocab_size_answer}")
    
    answer_model = TransformerSeq2Seq(vocab_size=vocab_size_answer).to(DEVICE)
    checkpoint_answer = torch.load("modelCA/best_model.pt", map_location=DEVICE)
    answer_model.load_state_dict(checkpoint_answer['model_state_dict'])
    answer_model.eval()
    print(f"✅ Answer model loaded (best val loss: {checkpoint_answer['val_loss']:.4f})")
    
    # Load Question Model (Answer -> Question)
    sp_question = spm.SentencePieceProcessor(model_file="modelAQ/tokenizer.model")
    vocab_size_question = sp_question.GetPieceSize()
    print(f"Loaded question tokenizer with vocab size: {vocab_size_question}")
    
    question_model = TransformerSeq2Seq(vocab_size=vocab_size_question).to(DEVICE)
    checkpoint_question = torch.load("modelAQ/best_model.pt", map_location=DEVICE)
    question_model.load_state_dict(checkpoint_question['model_state_dict'])
    question_model.eval()
    print(f"✅ Question model loaded (best val loss: {checkpoint_question['val_loss']:.4f})")
    
    print("\nStarting Quiz Application...")
    
    root = tk.Tk()
    app = QuizApplication(root, answer_model, question_model, sp_answer, sp_question)
    root.mainloop()


if __name__ == "__main__":
    main()
