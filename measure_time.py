from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from water_sort import *
import csv
import time

# Runtime analysis parameters
M = 1000  # number of states the algorithms are tested on
m = 10    # number of repeats for algorithm run

# State parameters
bin_volume = 4
filled_bins = 4
empty_bins = 2
colors = 4

# Results dataframe columns
data_columns = ["number", "bin_volume", "filled_bins", "empty_bins", "colors",
                "dfs_tree_time", "dfs_tree_sol", "dfs_tree_expanded",
                "dfs_graph_time", "dfs_graph_sol", "dfs_graph_expanded",
                "bfs_tree_time", "bfs_tree_sol", "bfs_tree_expanded",
                "bfs_graph_time", "bfs_graph_sol", "bfs_graph_expanded",
                "a_star_tree_h1_time", "a_star_tree_h1_sol", "a_star_tree_h1_expanded",
                "a_star_graph_h1_time", "a_star_graph_h1_sol", "a_star_graph_h1_expanded",
                "a_star_tree_h2_time", "a_star_tree_h2_sol", "a_star_tree_h2_expanded",
                "a_star_graph_h2_time", "a_star_graph_h2_sol", "a_star_graph_h2_expanded",
                "a_star_tree_h3_time", "a_star_tree_h3_sol", "a_star_tree_h3_expanded",
                "a_star_graph_h3_time", "a_star_graph_h3_sol", "a_star_graph_h3_expanded",
                "random_walk_time", "random_walk_sol", "random_walk_restart",
                "hill_sideways_h1_time", "hill_sideways_h1_sol",
                "hill_no_sideways_h1_time", "hill_no_sideways_h1_sol",
                "hill_sideways_h2_time", "hill_sideways_h2_sol",
                "hill_no_sideways_h2_time", "hill_no_sideways_h2_sol",
                "hill_sideways_h3_time", "hill_sideways_h3_sol",
                "hill_no_sideways_h3_time", "hill_no_sideways_h3_sol"
                ]

# Current datetime, for the csv file name
current = datetime.now(timezone(timedelta(hours=4))).strftime("%Y-%m-%d_%H-%M-%S")

# Open (create) the csv file and get the writer
f = open(f"results_{current}.csv", 'w')
writer = csv.writer(f)

# Write the column names to the csv
writer.writerow(data_columns)

# For M randomly generated states
for i in tqdm(range(M)):
    # Get the initial state
    initial_state = Game_State(bin_volume=bin_volume, filled_bins=filled_bins, empty_bins=empty_bins, colors=colors)

    start_time = time.time()
    for _ in range(m):
        # DFS Tree
        s, dfs_tree_expanded = DFS(initial_state, graph=False, test_mode=True)
    end_time = time.time()
    dfs_tree_sol = len(s) if s else None

    dfs_tree_time = (end_time - start_time)/m

    start_time = time.time()
    for _ in range(m):
        # DFS Graph
        s, dfs_graph_expanded = DFS(initial_state, graph=True, test_mode=True)
    end_time = time.time()
    dfs_graph_sol = len(s) if s else None

    dfs_graph_time = (end_time - start_time)/m

    start_time = time.time()
    for _ in range(m):
        # BFS Tree
        s, bfs_tree_expanded = BFS(initial_state, graph=False, test_mode=True)
    end_time = time.time()
    bfs_tree_sol = len(s) if s else None

    bfs_tree_time = (end_time - start_time)/m

    start_time = time.time()
    for _ in range(m):
        # BFS Graph
        s, bfs_graph_expanded = BFS(initial_state, graph=True, test_mode=True)
    end_time = time.time()
    bfs_graph_sol = len(s) if s else None

    bfs_graph_time = (end_time - start_time)/m

    start_time = time.time()

    for _ in range(m):
        # A* H1 Tree
        s, a_star_tree_h1_expanded = A_star(initial_state, heuristic=heuristic1, graph=False, test_mode=True)
    end_time = time.time()
    a_star_tree_h1_sol = len(s) if s else None

    a_star_tree_h1_time = (end_time - start_time)/m

    start_time = time.time()

    for _ in range(m):
        # A* H1 Graph
        s, a_star_graph_h1_expanded = A_star(initial_state, heuristic=heuristic1, graph=True, test_mode=True)
    end_time = time.time()
    a_star_graph_h1_sol = len(s) if s else None

    a_star_graph_h1_time = (end_time - start_time)/m

    start_time = time.time()
    for _ in range(m):
        # A* H2 Tree
        s, a_star_tree_h2_expanded = A_star(initial_state, heuristic=heuristic2, graph=False, test_mode=True)
    end_time = time.time()
    a_star_tree_h2_sol = len(s) if s else None

    a_star_tree_h2_time = (end_time - start_time)/m

    start_time = time.time()
    for _ in range(m):
        # A* H2 Graph
        s, a_star_graph_h2_expanded = A_star(initial_state, heuristic=heuristic2, graph=True, test_mode=True)
    end_time = time.time()
    a_star_graph_h2_sol = len(s) if s else None

    a_star_graph_h2_time = end_time - start_time

    start_time = time.time()
    for _ in range(m):
        # A* H3 Tree
        a_star_tree_h3_sol, a_star_tree_h3_expanded = A_star(initial_state, heuristic=heuristic3, graph=False, test_mode=True)
    end_time = time.time()
    a_star_tree_h3_sol = len(s) if s else None

    a_star_tree_h3_time = (end_time - start_time)/m

    start_time = time.time()
    for _ in range(m):
        # A* H3 Graph
        s, a_star_graph_h3_expanded = A_star(initial_state, heuristic=heuristic3, graph=True, test_mode=True)
    end_time = time.time()
    a_star_graph_h3_sol = len(s) if s else None

    a_star_graph_h3_time = (end_time - start_time)/m

    start_time = time.time()
    for _ in range(m):
        # Random Walk
        s, random_walk_restart = random_walk(initial_state, test_mode=True)
    end_time = time.time()
    random_walk_sol = len(s) if s else None

    random_walk_time = (end_time - start_time)/m

    start_time = time.time()
    for _ in range(m):
        # Hill Climbing H1 Sideways False
        s = hill_climbing(initial_state, heuristic=heuristic1, sideways=False)
    end_time = time.time()
    hill_sideways_h1_sol = len(s) if s else None

    hill_sideways_h1_time = (end_time - start_time)/m

    start_time = time.time()
    for _ in range(m):
        # Hill Climbing H1 Sideways True
        s = hill_climbing(initial_state, heuristic=heuristic1, sideways=True)
    end_time = time.time()
    hill_no_sideways_h1_sol = len(s) if s else None

    hill_no_sideways_h1_time = (end_time - start_time)/m

    start_time = time.time()
    for _ in range(m):
        # Hill Climbing H2 Sideways False
        s = hill_climbing(initial_state, heuristic=heuristic2, sideways=False)
    end_time = time.time()
    hill_sideways_h2_sol = len(s) if s else None

    hill_sideways_h2_time = (end_time - start_time)/m

    start_time = time.time()
    for _ in range(m):
        # Hill Climbing H2 Sideways True
        s = hill_climbing(initial_state, heuristic=heuristic2, sideways=True)
    end_time = time.time()
    hill_no_sideways_h2_sol = len(s) if s else None

    hill_no_sideways_h2_time = (end_time - start_time)/m

    start_time = time.time()
    for _ in range(m):
        # Hill Climbing H3 Sideways False
        s = hill_climbing(initial_state, heuristic=heuristic3, sideways=False)
    end_time = time.time()
    hill_sideways_h3_sol = len(s) if s else None

    hill_sideways_h3_time = (end_time - start_time)/m

    start_time = time.time()
    for _ in range(m):
        # Hill Climbing H3 Sideways True
        s = hill_climbing(initial_state, heuristic=heuristic3, sideways=True)
    end_time = time.time()
    hill_no_sideways_h3_sol = len(s) if s else None

    hill_no_sideways_h3_time = (end_time - start_time)/m

    # Write all results for this state to the csv file
    writer.writerow([i, bin_volume, filled_bins, empty_bins, colors,
                     dfs_tree_time, dfs_tree_sol, dfs_tree_expanded,
                     dfs_graph_time, dfs_graph_sol, dfs_graph_expanded,
                     bfs_tree_time, bfs_tree_sol, bfs_tree_expanded,
                     bfs_graph_time, bfs_graph_sol, bfs_graph_expanded,
                     a_star_tree_h1_time, a_star_tree_h1_sol, a_star_tree_h1_expanded,
                     a_star_graph_h1_time, a_star_graph_h1_sol, a_star_graph_h1_expanded,
                     a_star_tree_h2_time, a_star_tree_h2_sol, a_star_tree_h2_expanded,
                     a_star_graph_h2_time, a_star_graph_h2_sol, a_star_graph_h2_expanded,
                     a_star_tree_h3_time, a_star_tree_h3_sol, a_star_tree_h3_expanded,
                     a_star_graph_h3_time, a_star_graph_h3_sol, a_star_graph_h3_expanded,
                     random_walk_time, random_walk_sol, random_walk_restart,
                     hill_sideways_h1_time, hill_sideways_h1_sol,
                     hill_no_sideways_h1_time, hill_no_sideways_h1_sol,
                     hill_sideways_h2_time, hill_sideways_h2_sol,
                     hill_no_sideways_h2_time, hill_no_sideways_h2_sol,
                     hill_sideways_h3_time, hill_sideways_h3_sol,
                     hill_no_sideways_h3_time, hill_no_sideways_h3_sol
                     ])

# Close the csv file
f.close()