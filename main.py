from water_sort import *
import copy
import pygame
import subprocess

# Command to make an executable of the game
# pyinstaller --onefile main.py --collect-data assets/chars --collect-data assets/tiles --collect-data assets/fonts

pygame.init()

# Game parameters
WIDTH = 1250
HEIGHT = 600
BIN_HEIGHT = 200
BIN_WIDTH = 65
LEVEL1 = 300
LEVEL2 = 550
LEFT_PADDING = 400
MIDBIN_DISTANCE = 40

screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("Water Sort PyGame")
font = pygame.font.SysFont("poppins", 24)
fontS = pygame.font.SysFont("poppins", 14)
fps = 60
timer = pygame.time.Clock()
color_choices = ["red", "orange", "light blue", "dark blue", "dark green", "pink", "purple", "dark gray",
                 "brown", "light green", "yellow", "white"]
gs = None
initial_state = None
sol_initial = None
input_values = ["4", "4", "2", "4"]
input_box_active = [False] * 4
algo_box_active = [False] * 5
heur_box_active = [False] * 3
graph_box_active = [False] * 2
sideways_box_active = [False] * 2
solution = None
message = None
no_solution = False
prev_state = None
new_game = True
start_game = True
selected = False
select_rect = -1

def draw_tubes(gs):
    """
    Draws the tubes with the liquids in them.

    :param gs: the game state
    :return: the tube boxes (used for collision detection with the cursor)
    """
    tubes_num = len(gs.bins)
    tube_cols = gs.to_list()
    tube_boxes = []

    # Get tubes per row for the first and second rows
    if tubes_num % 2 == 0:
        tubes_per_row1 = tubes_per_row2 = tubes_num // 2
        offset = False
    else:
        tubes_per_row1 = tubes_num // 2 + 1
        tubes_per_row2 = tubes_num // 2
        offset = True

    # Get the sizes needed for drawing the tubes
    spacing = MIDBIN_DISTANCE + BIN_WIDTH
    water_unit_height = BIN_HEIGHT // gs.bin_volume
    additional_spacing = offset * spacing * 0.5
    additional_padding = (800 - tubes_per_row1 * BIN_WIDTH - (tubes_per_row1-1) * MIDBIN_DISTANCE) / 2
    total_padding = LEFT_PADDING + additional_padding

    # First row of tubes
    for i in range(tubes_per_row1):
        for j in range(len(tube_cols[i])):
            pygame.draw.rect(screen, color_choices[tube_cols[i][j]], [total_padding + spacing * i, LEVEL1 - (water_unit_height * (j+1)), BIN_WIDTH, water_unit_height], 0, 3)
        box = pygame.draw.rect(screen, "#4a83ed", [total_padding + spacing * i, LEVEL1-BIN_HEIGHT, BIN_WIDTH, BIN_HEIGHT], 5, 5)
        if select_rect == i:
            pygame.draw.rect(screen, "#2bed8c", [total_padding + spacing * i, LEVEL1-BIN_HEIGHT, BIN_WIDTH, BIN_HEIGHT], 3, 5)
        tube_boxes.append(box)

    # Second row of tubes
    for i in range(tubes_per_row2):
        for j in range(len(tube_cols[i + tubes_per_row1])):
            pygame.draw.rect(screen, color_choices[tube_cols[i + tubes_per_row1][j]],
                             [additional_spacing + total_padding + spacing * i, LEVEL2 - (water_unit_height * (j+1)), BIN_WIDTH, water_unit_height], 0, 3)
        box = pygame.draw.rect(screen, "#4a83ed", [additional_spacing + total_padding + spacing * i, LEVEL2-BIN_HEIGHT, BIN_WIDTH, BIN_HEIGHT], 5, 5)
        if select_rect == i + tubes_per_row1:
            pygame.draw.rect(screen, "#2bed8c", [additional_spacing + total_padding + spacing * i, LEVEL2-BIN_HEIGHT, BIN_WIDTH, BIN_HEIGHT], 3, 5)
        tube_boxes.append(box)
    return tube_boxes

def draw_input_area(input_box_active, input_values):
    """
    Draws the input area, where the game state properties are inputted.

    :param input_box_active: current selection
    :param input_values: current values
    :return: the location of boxes (used for collision detection with the cursor)
    """
    # Bin volume
    screen.blit(fontS.render("Bin Volume", True, "white"), (30, 80))
    fb_box = pygame.draw.rect(screen, "white", [30, 100, 100, 20], 0, 5)
    if input_box_active[0]:
        pygame.draw.rect(screen, "lightblue", [30, 100, 100, 20], 3, 5)
    screen.blit(fontS.render(input_values[0] if input_values[0] else "", True, "black"), (33, 100))

    # Filled bins
    screen.blit(fontS.render("Filled Bins", True, "white"), (30, 130))
    eb_box = pygame.draw.rect(screen, "white", [30, 150, 100, 20], 0, 5)
    if input_box_active[1]:
        pygame.draw.rect(screen, "lightblue", [30, 150, 100, 20], 3, 5)
    screen.blit(fontS.render(input_values[1] if input_values[1] else "", True, "black"), (33, 150))

    # Empty bins
    screen.blit(fontS.render("Empty Bins", True, "white"), (30, 180))
    bv_box = pygame.draw.rect(screen, "white", [30, 200, 100, 20], 0, 5)
    if input_box_active[2]:
        pygame.draw.rect(screen, "lightblue", [30, 200, 100, 20], 3, 5)
    screen.blit(fontS.render(input_values[2] if input_values[2] else "", True, "black"), (33, 200))

    # Colors
    screen.blit(fontS.render("Colors", True, "white"), (30, 230))
    colors_box = pygame.draw.rect(screen, "white", [30, 250, 100, 20], 0, 5)
    if input_box_active[3]:
        pygame.draw.rect(screen, "lightblue", [30, 250, 100, 20], 3, 5)
    screen.blit(fontS.render(input_values[3] if input_values[3] else "", True, "black"), (33, 250))

    # Re-generate button
    gen_box = pygame.draw.rect(screen, "lightgreen", [30, 280, 100, 20], 0, 5)
    screen.blit(fontS.render("Re-Generate", True, "black"), (33, 280))

    # Reset button
    res_box = pygame.draw.rect(screen, "orange", [30, 310, 100, 20], 0, 5)
    screen.blit(fontS.render("Reset", True, "black"), (33, 310))

    return [fb_box, eb_box, bv_box, colors_box], gen_box, res_box


def draw_algo_area(heuristic_required, graph_required, sideways_required):
    """
    Draws the algo selection area.

    :param heuristic_required: true if heuristic selection needs to be show
    :param graph_required: true if tree/graph selection needs to be show
    :param sideways_required: true if sideways selection needs to be show
    :return: the location of boxes (used for collision detection with the cursor)
    """
    # Initialize everything as None
    h1_box = None
    h2_box = None
    h3_box = None
    tree_box = None
    graph_box = None
    sideways_box1 = None
    sideways_box2 = None

    # Uninformed search strategies
    screen.blit(fontS.render("Uninformed Search", True, "white"), (150, 80))
    BFS_box = pygame.draw.circle(screen, "white", [160, 110], 5)
    if algo_box_active[0]:
        pygame.draw.circle(screen, "red", [160, 110], 4)
    screen.blit(fontS.render("BFS", True, "white"), (170, 100))

    DFS_box = pygame.draw.circle(screen, "white", [220, 110], 5)
    if algo_box_active[1]:
        pygame.draw.circle(screen, "red", [220, 110], 4)
    screen.blit(fontS.render("DFS", True, "white"), (230, 100))

    # Informed search strategies
    screen.blit(fontS.render("Informed Search", True, "white"), (150, 130))
    Astar_box = pygame.draw.circle(screen, "white", [160, 160], 5)
    if algo_box_active[2]:
        pygame.draw.circle(screen, "red", [160, 160], 4)
    screen.blit(fontS.render("A*", True, "white"), (170, 150))

    # Local search strategies
    screen.blit(fontS.render("Local Search", True, "white"), (150, 180))
    random_box = pygame.draw.circle(screen, "white", [160, 210], 5)
    if algo_box_active[3]:
        pygame.draw.circle(screen, "red", [160, 210], 4)
    screen.blit(fontS.render("Random Walk", True, "white"), (170, 200))

    hill_box = pygame.draw.circle(screen, "white", [160, 230], 5)
    if algo_box_active[4]:
        pygame.draw.circle(screen, "red", [160, 230], 4)
    screen.blit(fontS.render("Hill climbing", True, "white"), (170, 220))

    # Dividing line
    pygame.draw.line(screen, "white", (150, 245), (350, 245), 2)

    # Heuristic selection
    if heuristic_required:
        screen.blit(fontS.render("Heuristic | Evaluation Function", True, "white"), (150, 250))
        h1_box = pygame.draw.circle(screen, "white", [160, 280], 5)
        if heur_box_active[0]:
            pygame.draw.circle(screen, "red", [160, 280], 4)
        screen.blit(fontS.render("H1", True, "white"), (170, 270))

        h2_box = pygame.draw.circle(screen, "white", [200, 280], 5)
        if heur_box_active[1]:
            pygame.draw.circle(screen, "red", [200, 280], 4)
        screen.blit(fontS.render("H2", True, "white"), (210, 270))

        h3_box = pygame.draw.circle(screen, "white", [240, 280], 5)
        if heur_box_active[2]:
            pygame.draw.circle(screen, "red", [240, 280], 4)
        screen.blit(fontS.render("H3", True, "white"), (250, 270))

        if sideways_required:
            screen.blit(fontS.render("Sideways moves", True, "white"), (150, 300))
            sideways_box1 = pygame.draw.circle(screen, "white", [160, 330], 5)
            if sideways_box_active[0]:
                pygame.draw.circle(screen, "red", [160, 330], 4)
            screen.blit(fontS.render("Allowed", True, "white"), (170, 320))

            sideways_box2 = pygame.draw.circle(screen, "white", [160, 350], 5)
            if sideways_box_active[1]:
                pygame.draw.circle(screen, "red", [160, 350], 4)
            screen.blit(fontS.render("Not allowed", True, "white"), (170, 340))

    # Graph selection
    if graph_required:
        extra = algo_box_active[2] * 50
        screen.blit(fontS.render("Tree | Graph Search", True, "white"), (150, extra+250))
        tree_box = pygame.draw.circle(screen, "white", [160, extra+280], 5)
        if graph_box_active[0]:
            pygame.draw.circle(screen, "red", [160, extra+280], 4)
        screen.blit(fontS.render("Tree", True, "white"), (170, extra+270))

        graph_box = pygame.draw.circle(screen, "white", [220, extra+280], 5)
        if graph_box_active[1]:
            pygame.draw.circle(screen, "red", [220, extra+280], 4)
        screen.blit(fontS.render("Graph", True, "white"), (230, extra+270))

    # Solve button
    solve_box = pygame.draw.rect(screen, "lightgreen", [150, 380, 150, 20], 0, 5)
    screen.blit(fontS.render("Solve", True, "black"), (153, 380))

    return [BFS_box, DFS_box, Astar_box, random_box, hill_box],\
        [h1_box, h2_box, h3_box],\
        [tree_box, graph_box],\
        [sideways_box1, sideways_box2],\
        solve_box


def draw_solution_area(solution, no_solution, message):
    """
    Draws the solution area.

    :param solution: current solution
    :param no_solution: true if no solution was found as a result of running an algorithm
    :param message: the message to display (if no solution)
    :return: the location of boxes (used for collision detection with the cursor)
    """
    # If no solution found
    if no_solution:
        if message == "solution":
            pygame.draw.rect(screen, "white", [30, 470, 180, 20], 0, 5)
            screen.blit(fontS.render(f"No solution found.", True, "black"), (33, 470))
        else:
            pygame.draw.rect(screen, "white", [30, 470, 180, 20], 0, 5)
            screen.blit(fontS.render(f"{message}", True, "black"), (33, 470))

    # If solution found
    if solution:
        pygame.draw.rect(screen, "white", [30, 470, 180, 20], 0, 5)
        screen.blit(fontS.render(f"Solution length: {len(solution)}", True, "black"), (33, 470))

        copy_box = pygame.draw.rect(screen, "grey", [230, 470, 80, 20], 0, 5)
        screen.blit(fontS.render("Copy", True, "black"), (233, 470))

        prev_box = pygame.draw.rect(screen, "orange", [30, 500, 80, 20], 0, 5)
        screen.blit(fontS.render("Previous", True, "black"), (33, 500))

        next_box = pygame.draw.rect(screen, "lightblue", [130, 500, 80, 20], 0, 5)
        screen.blit(fontS.render("Next", True, "black"), (133, 500))

        goal_box = pygame.draw.rect(screen, "lightgreen", [230, 500, 80, 20], 0, 5)
        screen.blit(fontS.render("Goal", True, "black"), (233, 500))

        return prev_box, next_box, goal_box, copy_box
    return None, None, None, None

def try_move(state, src, dest):
    """
    Checks if there is a move from the given tube to the other given tube, returns true if succeeds.

    :param state: the state to check the move on
    :param src: the initial move
    :param dest: the destination move
    :return: true, if there is such a move, false otherwise
    """
    # Get the move if it exists
    move = state.any_legal_moves(src, dest)

    if move:
        state.move(move)
        return True
    return False

def copy2clip(txt):
    """
    Copies the given text to the clipboard.

    :param txt: the given text to copy to the clipboard
    :return: the result of the subprocess call
    """
    cmd='echo '+str(txt).strip()+'|clip'
    return subprocess.check_call(cmd, shell=True)

# check if every tube with colors is 4 long and all the same color. That"s how we win
def check_victory(gs):
    """
    Checks if the game was won (currently not used in the project).

    :param gs: the game state
    :return: true if game state has only full same-colored or empty bins
    """
    if not start_game:
        return gs.is_great_goal()

# Main game loop
run = True
while run:
    screen.fill("#222222")
    timer.tick(fps)

    # Generate the game state
    if start_game:
        if all(input_values) and all([x.isnumeric() for x in input_values]):
            gs = Game_State(*[int(x) for x in input_values])
            initial_state = copy.deepcopy(gs)
            start_game = False
        else:
            start_game = False
    else:
        tube_rects = draw_tubes(gs)

    # Get all boxes for further collision detection with the cursor
    input_boxes, gen_box, res_box = draw_input_area(input_box_active, input_values)
    heuristic_required = algo_box_active[2] or algo_box_active[4]
    graph_required = algo_box_active[0] or algo_box_active[1] or algo_box_active[2]
    sideways_required = algo_box_active[4]
    algo_boxes, heur_boxes, graph_boxes, sideways_boxes, solve_box = draw_algo_area(heuristic_required, graph_required, sideways_required)
    prev_box, next_box, goal_box, copy_box = draw_solution_area(solution, no_solution, message)

    # Check if game won (currently not used in the project)
    win = check_victory(gs)

    # Draw the title
    screen.blit(font.render("Water Sort Puzzle", True, "white"), (30, 30))

    # Check for events
    for event in pygame.event.get():
        # Game quit
        if event.type == pygame.QUIT:
            run = False

        # Keyboard interaction
        if event.type == pygame.KEYUP:

            for i in range(len(input_box_active)):
                # Inputting numbers in the input boxes
                if input_box_active[i]:
                    if event.unicode.isnumeric():
                        if input_values[i]:
                            input_values[i] += event.unicode
                        else:
                            input_values[i] = event.unicode
                    # Removing numbers from the input boxes
                    elif event.key == pygame.K_BACKSPACE:
                        input_values[i] = input_values[i][:-1]

        # Mouse interaction
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Clicking on input boxes
            for i in range(len(input_boxes)):
                if input_boxes[i].collidepoint(event.pos):
                    input_box_active = [False] * 4
                    input_box_active[i] = True

            # Clicking on algorithm selection buttons
            for i in range(len(algo_boxes)):
                if algo_boxes[i].collidepoint(event.pos):
                    algo_box_active = [False] * 5
                    algo_box_active[i] = True

            # If heuristic selection is shown
            if heuristic_required:
                # Clicking on heuristic selection buttons
                for i in range(len(heur_boxes)):
                    if heur_boxes[i].collidepoint(event.pos):
                        heur_box_active = [False] * 3
                        heur_box_active[i] = True
            else:
                heur_box_active = [False] * 3

            # If tree/graph selection is shown
            if graph_required:
                # Clicking on tree/graph selection buttons
                for i in range(len(graph_boxes)):
                    if graph_boxes[i].collidepoint(event.pos):
                        graph_box_active = [False] * 2
                        graph_box_active[i] = True
            else:
                graph_box_active = [False] * 2

            # If sideways selection is shown
            if sideways_required:
                # Clicking on sideways selection buttons
                for i in range(len(sideways_boxes)):
                    if sideways_boxes[i].collidepoint(event.pos):
                        sideways_box_active = [False] * 2
                        sideways_box_active[i] = True
            else:
                sideways_box_active = [False] * 2

            # Clicking on re-generate button
            if gen_box.collidepoint(event.pos):
                start_game = True
                solution = None
                current_move = -1
                solution_states = []

            # Clicking on reset button
            if res_box.collidepoint(event.pos):
                gs = copy.deepcopy(initial_state)
                current_move = -1
                solution_states = []

            # If the solution area should be shown
            if solution:
                # Clicking on the copy box
                if copy_box.collidepoint(event.pos):
                    copy2clip(solution)

                # Clicking on the previous box
                if prev_box.collidepoint(event.pos):
                    if current_move == 0:
                        prev_state, gs = None, copy.deepcopy(sol_initial)
                        solution_states.pop()
                        current_move -= 1
                    if current_move > 0:
                        prev_state, gs = solution_states[current_move-1], prev_state
                        solution_states.pop()
                        current_move -= 1

                # Clicking on the next box
                if next_box.collidepoint(event.pos):
                    if current_move == -1:
                        if gs != sol_initial:
                            gs = copy.deepcopy(sol_initial)
                        else:
                            solution_states.append(gs)
                            prev_state = gs
                            gs = gs.move(solution[current_move + 1], inplace=False)
                            current_move += 1
                    elif current_move < len(solution)-1:
                        solution_states.append(gs)
                        prev_state = gs
                        gs = gs.move(solution[current_move+1], inplace=False)
                        current_move += 1

                # Clicking on the goal box
                if goal_box.collidepoint(event.pos):
                    while current_move < len(solution)-1:
                        solution_states.append(gs)
                        prev_state = gs
                        gs = gs.move(solution[current_move+1], inplace=False)
                        current_move += 1

            # Clicking on the solve box
            if solve_box.collidepoint(event.pos):
                # gs = copy.deepcopy(initial_state)
                sol_initial = copy.deepcopy(gs)
                solution, message = solve(gs, algo_box_active, heur_box_active, graph_box_active, sideways_box_active)
                if not solution:
                    no_solution = True
                else:
                    no_solution = False
                solution_states = []
                current_move = -1
                print(solution)

            # Clicking on the tubes (for manual gameplay)
            if not selected:
                for item in range(len(tube_rects)):
                    if tube_rects[item].collidepoint(event.pos):
                        selected = True
                        select_rect = item
            else:
                for item in range(len(tube_rects)):
                    if tube_rects[item].collidepoint(event.pos):
                        dest_rect = item
                        if try_move(gs, select_rect, dest_rect) and solution:
                            solution_states = []
                            current_move = -1
                        selected = False
                        select_rect = 100

    pygame.display.flip()
pygame.quit()