import customtkinter as ctk
from tkinter import messagebox
from ui import backend

# Set theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class QuizApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Quiz Generator")
        
        # Make window resizable with minimum dimensions
        self.root.minsize(900, 600)
        
        # Start with a good default size but allow resizing
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(1100, int(screen_width * 0.8))
        window_height = min(750, int(screen_height * 0.85))
        
        # Center the window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # State
        self.current_category = None
        self.current_answer = None
        self.current_question = None
        
        # Easter egg state - cheat code
        self.k_press_count = 0
        self.k_press_timer = None
        self.answer_tooltip = None
        
        # Categories
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
        
        # Bind keyboard shortcut for easter egg
        self.root.bind('<KeyPress-k>', self.on_k_press)
        self.root.bind('<KeyPress-K>', self.on_k_press)
    
    def setup_ui(self):
        # Main scrollable container with auto-hide scrollbar
        main_scroll = ctk.CTkScrollableFrame(
            self.root, 
            fg_color="transparent",
            scrollbar_button_color="gray30",
            scrollbar_button_hover_color="gray40"
        )
        main_scroll.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Header
        header_frame = ctk.CTkFrame(main_scroll, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 30))
        
        title_label = ctk.CTkLabel(
            header_frame,
            text="AI Quiz Generator",
            font=ctk.CTkFont(family="Helvetica", size=42, weight="bold"),
            text_color=("#1f6aa5", "#4a9eff")
        )
        title_label.pack()
        
        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="Powered by Neural Networks",
            font=ctk.CTkFont(size=16),
            text_color="gray60"
        )
        subtitle_label.pack(pady=(5, 0))
        
        # Content area
        self.content_frame = ctk.CTkFrame(main_scroll, corner_radius=20)
        self.content_frame.pack(fill="both", expand=True)
        
        # Create frames
        self.category_frame = self.create_category_frame()
        self.question_frame = self.create_question_frame()
        self.result_frame = self.create_result_frame()
        
        # Show category frame initially
        self.show_category_frame()
        
        # Status label at bottom
        self.status_label = ctk.CTkLabel(
            main_scroll,
            text="",
            font=ctk.CTkFont(size=14),
            text_color="gray50"
        )
        self.status_label.pack(pady=(15, 0))
    
    def create_category_frame(self):
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        
        # Title
        title = ctk.CTkLabel(
            frame,
            text="Choose Your Challenge",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=("#1f6aa5", "#4a9eff")
        )
        title.pack(pady=(60, 10))
        
        subtitle = ctk.CTkLabel(
            frame,
            text="Select a category to begin your AI-generated quiz",
            font=ctk.CTkFont(size=16),
            text_color="gray60"
        )
        subtitle.pack(pady=(0, 50))
        
        # Category dropdown
        self.category_var = ctk.StringVar(value="Select a category...")
        
        self.category_dropdown = ctk.CTkComboBox(
            frame,
            values=self.categories,
            variable=self.category_var,
            width=400,
            height=50,
            font=ctk.CTkFont(size=16),
            dropdown_font=ctk.CTkFont(size=14),
            corner_radius=12,
            border_width=2,
            button_color=("#1f6aa5", "#4a9eff"),
            button_hover_color=("#174f7a", "#3d8ae6")
        )
        self.category_dropdown.pack(pady=30)
        
        # Generate button
        generate_btn = ctk.CTkButton(
            frame,
            text="Generate Question",
            command=self.generate_question,
            width=280,
            height=60,
            font=ctk.CTkFont(size=18, weight="bold"),
            corner_radius=30,
            fg_color=("#1f6aa5", "#4a9eff"),
            hover_color=("#174f7a", "#3d8ae6")
        )
        generate_btn.pack(pady=40)
        
        return frame
    
    def create_question_frame(self):
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        
        # Title
        title = ctk.CTkLabel(
            frame,
            text="Your Question",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color=("#1f6aa5", "#4a9eff")
        )
        title.pack(pady=(40, 20))
        
        # Question display
        question_container = ctk.CTkFrame(frame, corner_radius=15)
        question_container.pack(pady=20, padx=40, fill="both", expand=True)
        
        self.question_text = ctk.CTkTextbox(
            question_container,
            font=ctk.CTkFont(size=15),
            wrap="word",
            corner_radius=10,
            border_width=2,
            border_color=("#1f6aa5", "#4a9eff"),
            height=150,
            state="disabled"  # Make it read-only
        )
        self.question_text.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Answer section
        answer_label = ctk.CTkLabel(
            frame,
            text="Your Answer",
            font=ctk.CTkFont(size=20, weight="bold"),
            anchor="w"
        )
        answer_label.pack(pady=(30, 10), padx=40, anchor="w")
        
        answer_container = ctk.CTkFrame(frame, corner_radius=15)
        answer_container.pack(pady=10, padx=40, fill="x")
        
        self.user_answer_text = ctk.CTkTextbox(
            answer_container,
            font=ctk.CTkFont(size=14),
            wrap="word",
            corner_radius=10,
            border_width=2,
            border_color="gray40",
            height=120
        )
        self.user_answer_text.pack(fill="x", padx=15, pady=15)
        
        # Submit button
        submit_btn = ctk.CTkButton(
            frame,
            text="Check Answer",
            command=self.check_answer,
            width=250,
            height=55,
            font=ctk.CTkFont(size=18, weight="bold"),
            corner_radius=27,
            fg_color=("#16a34a", "#22c55e"),
            hover_color=("#15803d", "#16a34a")
        )
        submit_btn.pack(pady=30)
        
        return frame
    
    def create_result_frame(self):
        frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        
        # Title
        title = ctk.CTkLabel(
            frame,
            text="Results",
            font=ctk.CTkFont(size=36, weight="bold"),
            text_color=("#1f6aa5", "#4a9eff")
        )
        title.pack(pady=(40, 30))
        
        # Score card
        score_card = ctk.CTkFrame(frame, corner_radius=20, fg_color=("#e0e0e0", "#2b2b2b"))
        score_card.pack(pady=20, padx=40, fill="x")
        
        score_inner = ctk.CTkFrame(score_card, fg_color="transparent")
        score_inner.pack(pady=30, padx=30)
        
        score_label_text = ctk.CTkLabel(
            score_inner,
            text="Similarity Score:",
            font=ctk.CTkFont(size=18),
            text_color="gray60"
        )
        score_label_text.pack()
        
        self.score_label = ctk.CTkLabel(
            score_inner,
            text="0%",
            font=ctk.CTkFont(size=56, weight="bold"),
            text_color=("#16a34a", "#22c55e")
        )
        self.score_label.pack(pady=(10, 5))
        
        self.feedback_label = ctk.CTkLabel(
            score_inner,
            text="",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="gray70"
        )
        self.feedback_label.pack()
        
        # Answers section
        answers_container = ctk.CTkFrame(frame, fg_color="transparent")
        answers_container.pack(pady=20, padx=40, fill="both", expand=True)
        
        # Correct answer
        correct_label = ctk.CTkLabel(
            answers_container,
            text="Correct Answer",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=("#16a34a", "#22c55e"),
            anchor="w"
        )
        correct_label.pack(pady=(10, 10), anchor="w")
        
        correct_container = ctk.CTkFrame(answers_container, corner_radius=12)
        correct_container.pack(pady=(0, 20), fill="x")
        
        self.correct_answer_text = ctk.CTkTextbox(
            correct_container,
            font=ctk.CTkFont(size=13),
            wrap="word",
            height=100,
            corner_radius=10,
            border_width=2,
            border_color=("#16a34a", "#22c55e"),
            state="disabled"  # Read-only
        )
        self.correct_answer_text.pack(fill="x", padx=12, pady=12)
        
        # User answer
        user_label = ctk.CTkLabel(
            answers_container,
            text="Your Answer",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=("#1f6aa5", "#4a9eff"),
            anchor="w"
        )
        user_label.pack(pady=(10, 10), anchor="w")
        
        user_container = ctk.CTkFrame(answers_container, corner_radius=12)
        user_container.pack(fill="x")
        
        self.user_answer_display = ctk.CTkTextbox(
            user_container,
            font=ctk.CTkFont(size=13),
            wrap="word",
            height=100,
            corner_radius=10,
            border_width=2,
            border_color=("#1f6aa5", "#4a9eff"),
            state="disabled"  # Read-only
        )
        self.user_answer_display.pack(fill="x", padx=12, pady=12)
        
        # Try again button
        try_again_btn = ctk.CTkButton(
            frame,
            text="Try Another Question",
            command=self.reset_quiz,
            width=280,
            height=55,
            font=ctk.CTkFont(size=18, weight="bold"),
            corner_radius=27,
            fg_color=("#dc2626", "#ef4444"),
            hover_color=("#b91c1c", "#dc2626")
        )
        try_again_btn.pack(pady=30)
        
        return frame
    
    def show_category_frame(self):
        self.question_frame.pack_forget()
        self.result_frame.pack_forget()
        self.category_frame.pack(fill="both", expand=True)
    
    def show_question_frame(self):
        self.category_frame.pack_forget()
        self.result_frame.pack_forget()
        self.question_frame.pack(fill="both", expand=True)
    
    def show_result_frame(self):
        self.category_frame.pack_forget()
        self.question_frame.pack_forget()
        self.result_frame.pack(fill="both", expand=True)
    
    def generate_question(self):
        category = self.category_var.get()
        if category == "Select a category..." or not category:
            messagebox.showwarning("Warning", "Please select a category!")
            return
        
        # Create loading overlay
        loading_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color=("#f0f0f0", "#1a1a1a"),
            corner_radius=20
        )
        loading_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.8, relheight=0.4)
        
        loading_title = ctk.CTkLabel(
            loading_frame,
            text="Generating Your Quiz",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=("#1f6aa5", "#4a9eff")
        )
        loading_title.pack(pady=(40, 20))
        
        # Progress bar
        progress_bar = ctk.CTkProgressBar(
            loading_frame,
            width=400,
            height=20,
            corner_radius=10,
            progress_color=("#1f6aa5", "#4a9eff")
        )
        progress_bar.pack(pady=20)
        progress_bar.set(0)
        
        # Status text
        status_text = ctk.CTkLabel(
            loading_frame,
            text="",
            font=ctk.CTkFont(size=16),
            text_color="gray60"
        )
        status_text.pack(pady=10)
        
        self.root.update()
        
        try:
            # Step 1: Prepare generation
            status_text.configure(text="Analyzing category...")
            progress_bar.set(0.2)
            self.root.update()
            
            status_text.configure(text="Generating question from neural network...")
            progress_bar.set(0.4)
            self.root.update()
            
            # Use backend to generate question and answer
            self.current_question, self.current_answer = backend.generate_question(
                category=category,
                temperature=0.7
            )
            
            progress_bar.set(0.9)
            status_text.configure(text="Finalizing...")
            self.root.update()
            
            self.current_category = category
            
            # Display question
            self.question_text.configure(state="normal")
            self.question_text.delete("1.0", "end")
            self.question_text.insert("1.0", self.current_question)
            self.question_text.configure(state="disabled")
            
            progress_bar.set(1.0)
            self.root.update()
            
            # Remove loading overlay
            loading_frame.destroy()
            
            # Switch to question frame
            self.show_question_frame()
            
            self.status_label.configure(text="")
            
        except Exception as e:
            loading_frame.destroy()
            messagebox.showerror("Error", f"Failed to generate question: {str(e)}")
            self.status_label.configure(text="")
    
    def check_answer(self):
        user_answer = self.user_answer_text.get("1.0", "end").strip()
        
        if not user_answer:
            messagebox.showwarning("Warning", "Please enter your answer!")
            return
        
        # Calculate similarity score using backend
        score = backend.calculate_similarity(user_answer, self.current_answer)
        
        # Update score display
        self.score_label.configure(text=f"{score}%")
        
        # Color code the score
        if score >= 80:
            self.score_label.configure(text_color=("#16a34a", "#22c55e"))
            feedback = "Excellent! Outstanding work!"
        elif score >= 60:
            self.score_label.configure(text_color=("#eab308", "#fbbf24"))
            feedback = "Good effort! You're on the right track!"
        else:
            self.score_label.configure(text_color=("#dc2626", "#ef4444"))
            feedback = "Keep practicing! You'll get there!"
        
        self.feedback_label.configure(text=feedback)
        
        # Display correct answer
        self.correct_answer_text.configure(state="normal")
        self.correct_answer_text.delete("1.0", "end")
        self.correct_answer_text.insert("1.0", self.current_answer)
        self.correct_answer_text.configure(state="disabled")
        
        # Display user answer
        self.user_answer_display.configure(state="normal")
        self.user_answer_display.delete("1.0", "end")
        self.user_answer_display.insert("1.0", user_answer)
        self.user_answer_display.configure(state="disabled")
        
        # Switch to result frame
        self.show_result_frame()
    
    def reset_quiz(self):
        # Clear all fields
        self.category_var.set("Select a category...")
        self.user_answer_text.delete("1.0", "end")
        self.current_category = None
        self.current_answer = None
        self.current_question = None
        
        # Reset easter egg state
        self.k_press_count = 0
        if self.k_press_timer:
            self.root.after_cancel(self.k_press_timer)
            self.k_press_timer = None
        if self.answer_tooltip:
            self.answer_tooltip.destroy()
            self.answer_tooltip = None
        
        # Switch back to category frame
        self.show_category_frame()
    
    def on_k_press(self, event):
        """Easter egg: Press K 3 times to reveal answer"""
        # Only works when on question frame and have an answer
        if not self.current_answer or not self.question_frame.winfo_ismapped():
            return
        
        # Increment counter
        self.k_press_count += 1
        
        # Cancel previous timer if exists
        if self.k_press_timer:
            self.root.after_cancel(self.k_press_timer)
        
        # Reset counter after 2 seconds
        self.k_press_timer = self.root.after(2000, self.reset_k_counter)
        
        # If pressed 3 times, show the answer
        if self.k_press_count >= 3:
            self.show_answer_tooltip()
            self.k_press_count = 0
    
    def reset_k_counter(self):
        """Reset the K press counter"""
        self.k_press_count = 0
        self.k_press_timer = None
    
    def show_answer_tooltip(self):
        """Show a tooltip with the correct answer"""
        # Remove existing tooltip if any
        if self.answer_tooltip:
            self.answer_tooltip.destroy()
        
        # Create tooltip
        self.answer_tooltip = ctk.CTkToplevel(self.root)
        self.answer_tooltip.title("Psst... Here's the answer!")
        
        # Remove window decorations for a cleaner look
        self.answer_tooltip.overrideredirect(True)
        
        # Position near the center but slightly offset
        x = self.root.winfo_x() + self.root.winfo_width() // 2 - 200
        y = self.root.winfo_y() + self.root.winfo_height() // 2 - 100
        self.answer_tooltip.geometry(f"400x200+{x}+{y}")
        
        # Make it stay on top
        self.answer_tooltip.attributes('-topmost', True)
        
        # Content frame
        content = ctk.CTkFrame(
            self.answer_tooltip,
            corner_radius=15,
            fg_color=("#16a34a", "#22c55e"),
            border_width=3,
            border_color=("#15803d", "#16a34a")
        )
        content.pack(fill="both", expand=True, padx=3, pady=3)
        
        # Title
        title = ctk.CTkLabel(
            content,
            text="Secret Answer Revealed!",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="white"
        )
        title.pack(pady=(15, 10))
        
        # Answer text
        answer_box = ctk.CTkTextbox(
            content,
            font=ctk.CTkFont(size=13),
            wrap="word",
            height=80,
            fg_color=("#f0fdf4", "#1a2e1a"),
            text_color=("black", "white"),
            corner_radius=8
        )
        answer_box.pack(padx=20, pady=10, fill="both", expand=True)
        answer_box.insert("1.0", self.current_answer)
        answer_box.configure(state="disabled")
        
        # Close button
        close_btn = ctk.CTkButton(
            content,
            text="Got it!",
            command=self.answer_tooltip.destroy,
            width=100,
            height=30,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="white",
            text_color=("#16a34a", "#22c55e"),
            hover_color=("#f0f0f0", "#e0e0e0"),
            corner_radius=15
        )
        close_btn.pack(pady=(0, 15))
        
        # Auto-close after 10 seconds
        self.root.after(10000, lambda: self.answer_tooltip.destroy() if self.answer_tooltip else None)


def main():
    root = ctk.CTk()
    app = QuizApplication(root)
    root.mainloop()


if __name__ == "__main__":
    main()