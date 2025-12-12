import dearpygui.dearpygui as dpg
import threading
from backend import generate_question, calculate_similarity

state = {
    "correct_answer": ""
}

CATEGORIES = [
    "n-queens",
    "generalised_hanoi",
    "graph_coloring",
    "knights_tour",
    "csp_backtracking",
    "minmax_alphabeta",
    "game_theory"
]

# ----------------------------------
# CALLBACKS
# ----------------------------------
def generate_cb():
    category = dpg.get_value("category")
    temp = dpg.get_value("difficulty")

    if not category:
        return

    def task():
        q, a = generate_question(category, temp)
        state["correct_answer"] = a
        dpg.set_value("question", q)
        dpg.set_value("answer", "")
        dpg.set_value("score", 0)
        dpg.set_value("feedback", "")

    threading.Thread(target=task, daemon=True).start()


def submit_cb():
    user = dpg.get_value("answer")
    if not user:
        return

    score = calculate_similarity(user, state["correct_answer"])
    dpg.set_value("score", score / 100)
    dpg.set_value("feedback", f"{score}% similarity")
    dpg.set_value("correct", state["correct_answer"])

# ----------------------------------
# UI
# ----------------------------------
dpg.create_context()

with dpg.window(label="AI Quiz Generator", width=1100, height=650):
    with dpg.group(horizontal=True):

        with dpg.child_window(width=260):
            dpg.add_text("⚙️ Settings")
            dpg.add_combo(CATEGORIES, tag="category")
            dpg.add_slider_float(label="Difficulty", min_value=0.3, max_value=1.2, default_value=0.7, tag="difficulty")
            dpg.add_button(label="Generate Question", callback=generate_cb)

        with dpg.child_window(width=520):
            dpg.add_text("❓ Question")
            dpg.add_input_text(multiline=True, readonly=True, height=140, tag="question")
            dpg.add_text("✍️ Your Answer")
            dpg.add_input_text(multiline=True, height=140, tag="answer")
            dpg.add_button(label="Submit Answer", callback=submit_cb)

        with dpg.child_window():
            dpg.add_text("📊 Result")
            dpg.add_progress_bar(tag="score")
            dpg.add_text("", tag="feedback")
            dpg.add_text("✅ Correct Answer")
            dpg.add_input_text(multiline=True, readonly=True, height=140, tag="correct")

dpg.create_viewport(title="AI Quiz Generator", width=1100, height=650)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
