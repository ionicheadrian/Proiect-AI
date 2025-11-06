import json, random, itertools

TOTAL_SAMPLES = 30000
OUTPUT_FILE = "ai_dataset.json"
random.seed(42)

# ---------- QUESTION TEMPLATES ----------
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

# ---------- HELPERS ----------
def random_subset(lst, k):
    return random.sample(lst, min(k, len(lst)))

# ---------- GENERATORS ----------
def gen_n_queens():
    size = random.choice([4, 6, 8, 10, 12, 15, 20])
    partial = [(r, random.randint(1, size)) for r in random_subset(range(1, size+1), random.randint(1, min(4, size)))]
    question = random.choice(TEMPLATES["n-queens"]).format(size=size, assignment=partial)
    strategy = (
        "Backtracking with Forward Checking (FC) and Minimum Remaining Values (MRV)"
        if size <= 12 else
        "Local Search (Hill-Climbing with random restarts)"
    )
    return {"problem_type": "n-queens", "question_text": question, "answer_text": strategy}

def gen_graph_coloring():
    num_vertices = random.randint(4, 10)
    edges = [(i, j) for i, j in itertools.combinations(range(1, num_vertices+1), 2) if random.random() < 0.3]
    num_colors = random.choice([2, 3, 4, 5])
    question = random.choice(TEMPLATES["graph_coloring"]).format(
        num_vertices=num_vertices, edges=edges, num_colors=num_colors
    )
    answer = (
        "Backtracking with MRV and Forward Checking" if num_vertices <= 6 else
        "DSATUR heuristic (Degree of Saturation)" if num_vertices <= 10 else
        "Greedy coloring (Largest degree first)"
    )
    return {"problem_type": "graph_coloring", "question_text": question, "answer_text": answer}

def gen_generalised_hanoi():
    num_disks = random.randint(3, 12)
    num_pegs = random.choice([3, 4, 5])
    question = random.choice(TEMPLATES["generalised_hanoi"]).format(num_disks=num_disks, num_pegs=num_pegs)
    answer = "Recursive optimal 3-peg algorithm" if num_pegs == 3 else "Frame–Stewart heuristic"
    return {"problem_type": "generalised_hanoi", "question_text": question, "answer_text": answer}

def gen_knights_tour():
    board_size = random.choice([5, 6, 7, 8])
    question = random.choice(TEMPLATES["knights_tour"]).format(board_size=board_size)
    answer = "Warnsdorff's heuristic rule" if board_size <= 8 else "Backtracking with pruning"
    return {"problem_type": "knights_tour", "question_text": question, "answer_text": answer}

def gen_csp_backtracking():
    vars = [f"X{i}" for i in range(1, random.randint(3, 6))]
    domain = {v: [1, 2, 3] for v in vars}
    constraints = [f"{a}!={b}" for a, b in itertools.combinations(vars, 2) if random.random() < 0.4]
    partial = {random.choice(vars): random.choice([1, 2, 3])}
    question = random.choice(TEMPLATES["csp_backtracking"]).format(
        variables=vars, domain=domain, constraints=constraints, partial_assignment=partial
    )
    assignment = partial.copy()
    for v in vars:
        if v not in assignment:
            allowed = [val for val in [1, 2, 3] if val not in assignment.values()]
            # fallback dacă lista e goală (toate valorile folosite)
            if not allowed:
                allowed = [1, 2, 3]
            assignment[v] = random.choice(allowed)
    return {
        "problem_type": "csp_backtracking",
        "question_text": question,
        "answer_text": f"Complete assignment: {assignment}"
    }

def minimax_value(leaves, depth, maximizing=True):
    if depth == 1:
        return max(leaves) if maximizing else min(leaves)
    group_size = len(leaves) // 2
    sub_values = [
        minimax_value(leaves[i:i+group_size], depth-1, not maximizing)
        for i in range(0, len(leaves), group_size)
    ]
    return max(sub_values) if maximizing else min(sub_values)

def gen_minmax_alphabeta():
    depth = random.choice([2, 3])
    leaves = [random.randint(-10, 10) for _ in range(2 ** depth)]
    question = random.choice(TEMPLATES["minmax_alphabeta"]).format(depth=depth, leaves=leaves)
    value = minimax_value(leaves, depth)
    answer = f"Root value: {value} (Alpha-Beta would visit fewer than {len(leaves)} leaves)"
    return {"problem_type": "minmax_alphabeta", "question_text": question, "answer_text": answer}

# ---------- Game Theory (improved) ----------
def find_pure_nash(matrix):
    """Finds pure Nash equilibria for a 2x2 payoff matrix [(a11,a12),(a21,a22)] etc."""
    nash_eq = []
    rows, cols = len(matrix), len(matrix[0])
    for i in range(rows):
        for j in range(cols):
            player1_payoff = matrix[i][j][0]
            player2_payoff = matrix[i][j][1]
            best_for_p1 = all(player1_payoff >= matrix[r][j][0] for r in range(rows))
            best_for_p2 = all(player2_payoff >= matrix[i][c][1] for c in range(cols))
            if best_for_p1 and best_for_p2:
                nash_eq.append((i, j))
    return nash_eq

def gen_game_theory():
    # choose matrix size randomly (2x2, 2x3 or 3x3)
    rows, cols = random.choice([(2, 2), (2, 3), (3, 3)])
    
    # generate random integer payoffs for each player
    matrix = [[(random.randint(0, 5), random.randint(0, 5)) for _ in range(cols)] for _ in range(rows)]
    
    def find_pure_nash(mat):
        nash_eq = []
        r, c = len(mat), len(mat[0])
        for i in range(r):
            for j in range(c):
                p1_payoff = mat[i][j][0]
                p2_payoff = mat[i][j][1]
                best_for_p1 = all(p1_payoff >= mat[x][j][0] for x in range(r))
                best_for_p2 = all(p2_payoff >= mat[i][y][1] for y in range(c))
                if best_for_p1 and best_for_p2:
                    nash_eq.append((i, j))
        return nash_eq

    nash = find_pure_nash(matrix)
    question = random.choice(TEMPLATES["game_theory"]).format(matrix=matrix)
    
    if nash:
        eq_str = ", ".join([f"(row={i}, col={j}) payoff={matrix[i][j]}" for i, j in nash])
        answer = f"Pure Nash equilibria: {eq_str}."
    else:
        answer = "No pure Nash equilibrium exists."
    
    return {
        "problem_type": "game_theory",
        "question_text": question,
        "answer_text": answer
    }

# ---------- MAIN ----------
GEN_FUNCS = [
    gen_n_queens,
    gen_graph_coloring,
    gen_generalised_hanoi,
    gen_knights_tour,
    gen_csp_backtracking,
    gen_minmax_alphabeta,
    gen_game_theory
]

def main():
    per_type = TOTAL_SAMPLES // len(GEN_FUNCS)
    dataset = []
    for func in GEN_FUNCS:
        dataset.extend(func() for _ in range(per_type))
    random.shuffle(dataset)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"✅ Generated {len(dataset)} samples in '{OUTPUT_FILE}'")

if __name__ == "__main__":
    main()
