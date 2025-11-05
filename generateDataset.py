import json
import random
import itertools

OUTPUT_FILE = "qa_question_dataset.json"
NUM_EXAMPLES = 10000  # adjust for more data

# -------------------------------
# Helper functions to generate instances
# -------------------------------

def generate_n_queens_instance(min_size=4, max_size=12):
    size = random.randint(min_size, max_size)
    partial_assignment = {f"Q{i+1}": random.randint(1, size) for i in range(random.randint(0, size//2))}
    return size, partial_assignment

def generate_graph_coloring_instance():
    num_vertices = random.randint(4, 8)
    edges = list(itertools.combinations(range(1, num_vertices+1), 2))
    edges = random.sample(edges, random.randint(num_vertices, len(edges)))
    num_colors = random.randint(2, 4)
    return num_vertices, edges, num_colors

def generate_hanoi_instance():
    num_disks = random.randint(3, 6)
    num_pegs = random.randint(3, 5)
    return num_disks, num_pegs

def generate_knights_tour_instance():
    board_size = random.choice([5, 6, 8])
    return board_size

def generate_csp_instance():
    variables = [f"X{i+1}" for i in range(random.randint(3,5))]
    domain = [1, 2, 3]
    constraints = [f"{random.choice(variables)} != {random.choice(variables)}" for _ in range(len(variables))]
    partial_assignment = {random.choice(variables): random.choice(domain)}
    return variables, domain, constraints, partial_assignment

def generate_minmax_instance():
    depth = random.randint(2,3)
    leaves = [random.randint(0,10) for _ in range(2**depth)]
    return depth, leaves

def generate_game_theory_instance():
    size = random.randint(2,3)
    matrix = [[(random.randint(0,3), random.randint(0,3)) for _ in range(size)] for _ in range(size)]
    return matrix

# -------------------------------
# Templates for question generation
# -------------------------------

TEMPLATES = {
    "n-queens": [
        "For the {size}-Queens problem with partial assignment {assignment}, which strategy should be used to complete the solution?",
        "Given a {size}-Queens board and the current positions {assignment}, what is the next best move strategy?"
    ],
    "graph_coloring": [
        "For a graph with {num_vertices} vertices and edges {edges}, using {num_colors} colors, which strategy would you apply to assign colors efficiently?",
        "Given a graph with {num_vertices} nodes and {num_colors} colors, how would you assign colors to avoid conflicts?"
    ],
    "generalised_hanoi": [
        "For the Towers of Hanoi with {num_disks} disks and {num_pegs} pegs, which strategy minimizes the number of moves?",
        "Given {num_disks} disks and {num_pegs} pegs in Hanoi, what is the optimal solving strategy?"
    ],
    "knights_tour": [
        "For a {board_size}x{board_size} chessboard, which strategy is most suitable to generate a Knight's Tour?",
        "Given a {board_size}x{board_size} board, how would you complete a Knight's Tour efficiently?"
    ],
    "csp_backtracking": [
        "Given variables {variables}, domains {domain}, constraints {constraints}, and partial assignment {partial_assignment}, what are the next variable assignments using Backtracking?",
        "With variables {variables} and constraints {constraints}, how would you continue the assignment {partial_assignment}?"
    ],
    "minmax_alphabeta": [
        "For a MinMax tree of depth {depth} with leaf values {leaves}, what would be the root value using Alpha-Beta pruning?",
        "Given a MinMax tree with depth {depth} and leaves {leaves}, how many nodes are visited and what is the root value using Alpha-Beta?"
    ],
    "game_theory": [
        "For the normal-form game represented by the payoff matrix {matrix}, identify any pure Nash equilibria.",
        "Given the game matrix {matrix}, is there a pure Nash equilibrium, and if so, which one?"
    ]
}

# -------------------------------
# Generate dataset
# -------------------------------

dataset = []

for i in range(NUM_EXAMPLES):
    problem_type = random.choice(list(TEMPLATES.keys()))
    
    if problem_type == "n-queens":
        size, assignment = generate_n_queens_instance()
        template = random.choice(TEMPLATES["n-queens"])
        question = template.format(size=size, assignment=assignment)
    elif problem_type == "graph_coloring":
        num_vertices, edges, num_colors = generate_graph_coloring_instance()
        template = random.choice(TEMPLATES["graph_coloring"])
        question = template.format(num_vertices=num_vertices, edges=edges, num_colors=num_colors)
    elif problem_type == "generalised_hanoi":
        num_disks, num_pegs = generate_hanoi_instance()
        template = random.choice(TEMPLATES["generalised_hanoi"])
        question = template.format(num_disks=num_disks, num_pegs=num_pegs)
    elif problem_type == "knights_tour":
        board_size = generate_knights_tour_instance()
        template = random.choice(TEMPLATES["knights_tour"])
        question = template.format(board_size=board_size)
    elif problem_type == "csp_backtracking":
        variables, domain, constraints, partial_assignment = generate_csp_instance()
        template = random.choice(TEMPLATES["csp_backtracking"])
        question = template.format(
            variables=variables,
            domain=domain,
            constraints=constraints,
            partial_assignment=partial_assignment
        )
    elif problem_type == "minmax_alphabeta":
        depth, leaves = generate_minmax_instance()
        template = random.choice(TEMPLATES["minmax_alphabeta"])
        question = template.format(depth=depth, leaves=leaves)
    elif problem_type == "game_theory":
        matrix = generate_game_theory_instance()
        template = random.choice(TEMPLATES["game_theory"])
        question = template.format(matrix=matrix)
    
    dataset.append({
        "question_id": str(i+1),
        "problem_type": problem_type,
        "question_text": question,
        "answer_text": ""  # leave blank for seq2seq to generate
    })

# -------------------------------
# Save dataset
# -------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"✅ Generated {NUM_EXAMPLES} questions for seq2seq dataset: {OUTPUT_FILE}")
