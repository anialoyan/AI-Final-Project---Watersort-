from collections import deque
import copy
import heapq
import numpy as np


class Water_Unit:
    """
        A class used to represent a water unit.

        ...

        Attributes
        ----------
        color : int
            The liquid's color.
        volume : int
            The liquid's volume.

        Methods
        -------
        add(vol):
            Add the given volume to the water unit.
        take(vol):
            Takes the given volume from the water unit.
    """
    def __init__(self, color, volume=1):
        """
        Initializes the water unit.

        :param color: liquid's color
        :param volume: liquid's volume
        """
        self.color = color
        self.volume = volume

    def take(self, vol=1):
        """
        Takes the given volume from the water unit.

        :param vol: the volume to take
        """
        # If not enough volume in the water unit
        if vol > self.volume:
            raise Exception(f"Illegal argument: amount={vol} > volume={self.volume}")

        self.volume -= vol

    def add(self, vol=1):
        """
        Add the given volume to the water unit.

        :param vol: the volume to add
        """
        self.volume += vol

    def __str__(self):
        return f"{self.color}:{self.volume}L"


# Represents a tube, which can contain water units
class Bin:
    """
    A class used to represent a game state.

    ...

    Attributes
    ----------
    water : list of water units
        List of Bin objects representing the tubes.
    free : int
        The tube volume.

    Methods
    -------
    get_top():
        Returns the top water unit if the bin is not empty.
    get_bottom():
        Returns the bottom water unit if the bin is not empty.
    add(new_unit):
        Adds the new water unit to the bin.
    take(vol):
        Takes the given volume of water from the top of the bin.
    pop():
        Removes the topmost water unit. Updates the free volume.
    is_empty():
        Returns true if the bin is empty.
    is_complete():
        Returns true if the bin is complete (only one color).
    is_goal_bin():
        Returns true if the bin is a goal bin (empty or complete).
    is_full():
        Returns true if the bin is full.
    is_fully_complete():
        Returns true if the bin is fully complete (only one color and no free space).
    to_list():
        Converts the bin to a list. This is needed for the pygame.
    """
    def __init__(self, volume=4):
        """
        Initializes the bin.

        :param volume: the volume of the bin
        """
        self.water = []
        self.free = volume

    def get_top(self):
        """
        Returns the top water unit if the bin is not empty.

        :return: the topmost water unit
        """
        if self.is_empty():
            return None

        return self.water[-1]

    def get_bottom(self):
        """
        Returns the bottom water unit if the bin is not empty.

        :return: the bottommost water unit
        """
        if self.is_empty():
            return None

        return self.water[0]

    def add(self, new_unit):
        """
        Adds the new water unit to the bin.

        :param new_unit: the water unit
        """
        # If not enough room
        if new_unit.volume > self.free:
            raise Exception(f"Illegal argument: volume of new unit={new_unit.volume} > free volume={self.free}")

        # If bin empty
        if self.is_empty():
            self.water.append(new_unit)
        # If colors match
        elif self.water[-1].color == new_unit.color:
            self.water[-1].add(new_unit.volume)
        # Otherwise
        else:
            self.water.append(new_unit)

        # Subtract the volume added from free
        self.free -= new_unit.volume

    def take(self, vol):
        """
        Takes the given volume of water from the top of the bin.

        :param vol: the volume to take
        """
        # If bin empty
        if self.is_empty():
            raise Exception(f"Illegal argument: bin empty, can't take from it")

        # If the topmost water unit to be taken fully
        if self.water[-1].volume == vol:
            self.pop()
        # Otherwise
        else:
            # Take the water (partially)
            self.water[-1].take(vol)

            # Add the volume taken from free
            self.free += vol

    def pop(self):
        """
        Removes the topmost water unit. Updates the free volume.
        """
        # If bin empty
        if self.is_empty():
            raise Exception(f"Illegal argument: bin empty, can't pop from it")

        # Add the volume removed to free volume
        self.free += self.water[-1].volume

        # Remove the topmost water unit
        self.water.pop()

    def is_empty(self):
        """
        Returns true if the bin is empty.

        :return: true if bin is empty
        """
        return len(self.water) == 0  # empty

    def is_complete(self):
        """
        Returns true if the bin is complete (only one color).

        :return: true if bin is complete
        """
        return len(self.water) == 1  # only one water unit

    def is_goal_bin(self):
        """
        Returns true if the bin is a goal bin (empty or complete).

        :return: true if bin is a goal bin
        """
        return True if self.is_empty() or self.is_complete() else False # empty or complete

    def is_full(self):
        """
        Returns true if the bin is full.

        :return: true if bin is full
        """
        return self.free == 0  # no free space

    def is_fully_complete(self):
        """
        Returns true if the bin is fully complete (only one color and no free space).

        :return: true if bin is fully complete
        """
        return self.is_complete() and self.is_full()  # complete and full

    def to_list(self):
        """
        Converts the bin to a list. This is needed for the pygame.

        :return: list representation of the bin
        """
        l = []
        for wu in self.water:
            l.extend([wu.color] * wu.volume)
        return l

    def __str__(self):
        return "  ".join([str(wu) for wu in self.water])


# Represents a game state, that is, the collection of tubes with liquids
class Game_State:
    """
    A class used to represent a game state.

    ...

    Attributes
    ----------
    bins : list of bins
        List of Bin objects representing the tubes.
    bin_volume : int
        The tube volume.
    filled_bins : int
        Number of filled bins.
    empty_bins : int
        Number of empty bins.
    colors : int
        Number of colors.

    Methods
    -------
    get_legal_moves():
        Finds and returns a list of legal moves for this state (in increasing order).
    any_legal_moves(src, dest):
        Returns the move, if there is one from a given tube to another given tube, None otherwise.
    move(move, verbose=False, inplace=True):
        Makes the move and returns the new state, or updates this state.
    is_goal():
        Checks if state is goal and returns true if it is so.
    is_great_goal(self):
        Checks if state is goal and all non-empty tubes are full. Returns true if it is so.
    to_list():
        Converts the state to a list. This is needed for the pygame.
    """
    def __init__(self, bin_volume=4, filled_bins=4, empty_bins=2, colors=4, order=None):
        """
        Randomly initializes a state with given parameters.

        :param bin_volume: The tube volume.
        :param filled_bins: Number of filled bins.
        :param empty_bins: Number of empty bins.
        :param colors: Number of colors.
        :param order:  A list of length (filled_bins*bin_volume) with colors distinct values, representing an
        initial state. This si used for testing purposes only. The code does not check the validity of the given list.
        """

        # There should be at least as many filled_bins as colors
        if filled_bins < colors:
            raise Exception("Filled bins cannot be less than colors.")

        # Initialize the bins
        self.bins = [Bin(bin_volume) for _ in range(filled_bins + empty_bins)]

        # Store parameters describing the state
        self.bin_volume = bin_volume
        self.filled_bins = filled_bins
        self.empty_bins = empty_bins
        self.colors = colors

        # Does not check if order is correct | order used for testing purposes only
        # Randomly generate the game state if order not provided
        if not order:
            order = (list(range(colors))*(filled_bins//colors) + list(np.random.choice(colors, filled_bins%colors)))*bin_volume
            np.random.shuffle(order)

        # Create the bins using the order list
        for i in range(filled_bins):
            for color in order[i * bin_volume:(i + 1) * bin_volume]:
                self.bins[i].add(Water_Unit(color=color))

    def get_legal_moves(self):
        """
        Finds and returns a list of legal moves for this state (in increasing order).

        :return: A list of legal moves for this state.
        """

        # To store the moves
        moves = []

        # For each bin (initial)
        for i, bi in enumerate(self.bins):
            # For each other bin (target)
            for j, bj in enumerate(self.bins):
                # If initial bin is empty or fully complete, break
                if bi.is_empty() or bi.is_fully_complete():
                    break

                # If the same bin is selected for initial and target
                # or the target bin is full, continue
                if i == j or bj.is_full():
                    continue

                # Move is legal if colors match or target tube is empty, and there is enough room
                if bj.is_empty() or bi.get_top().color == bj.get_top().color:
                    moves.append((i, j, min(bj.free, bi.get_top().volume)))
        return moves

    def any_legal_moves(self, src, dest):
        """
        Returns the move, if there is one from a given tube to another given tube, None otherwise.

        :param src: the initial tube number
        :param dest: the target tube number
        :return: move, if there is one
        """
        # Get the tubes
        bi = self.bins[src]
        bj = self.bins[dest]

        # If initial bin is empty or fully complete, return None
        if bi.is_empty() or bi.is_fully_complete():
            return None

        # If the same bin is selected for initial and target
        # or the target bin is full, return None
        if src == dest or dest.is_full():
            return None

        # Move is legal if colors match or target tube is empty, and there is enough room
        if bj.is_empty() or bi.get_top().color == bj.get_top().color:
            if bj.free > 0:
                return (src, dest, min(bj.free, bi.get_top().volume))
        return None

    def move(self, move, verbose=False, inplace=True):
        """
        Makes the move and returns the new state, or updates this state.

        :param move: the move to make
        :param verbose: if true, prints the move; for debugging purposes
        :return: the result of the move or None if this state is changed
        """
        if verbose:
            print(f"{move}")

        # If inplace is false, make a copy of this state and perform the move on it
        state = self if inplace else copy.deepcopy(self)

        # Decompose the move
        i, j, vol = move

        # Get the top water unit
        top_i = state.bins[i].get_top()

        # Add the liquid to the target state
        state.bins[j].add(Water_Unit(color=top_i.color, volume=vol))

        # Take the liquid from the initial bin
        state.bins[i].take(vol)

        # If inplace is false, return the new state
        if not inplace:
            return state

    def is_goal(self):
        """
        Checks if state is goal and returns true if it is so.

        :return: true if this is a goal state
        """
        return all(b.is_goal_bin() for b in self.bins)

    def is_great_goal(self):
        """
        Checks if state is goal and all non-empty tubes are full. Returns true if it is so.

        :return: true if this is a great goal state
        """
        return self.is_goal() and sum(b.is_empty() for b in self.bins) == self.empty_bins

    def to_list(self):
        """
        Converts the state to a list. This is needed for the pygame.

        :return: list representation of the state
        """
        return [b.to_list() for b in self.bins]

    def __str__(self):
        return "".join(f"Bin {i}: {b}\n" for i, b in enumerate(self.bins))

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __hash__(self):
        return hash(self.__str__())


def do_random_move(state, verbose=False):
    """
    Selects a random move from the set of legal moves (if there are any) and performs it. Returns the resulting state.

    :param state: initial state
    :param verbose: if true, prints the move; for debugging purposes
    :return: the state resulting from performing a random action on the initial state
    """
    moves = state.get_legal_moves()

    if len(moves) == 0:
        return None, None

    else:
        random_move = moves[np.random.randint(len(moves))]

        if verbose:
            print(random_move)

        state = state.move(random_move, verbose, inplace=False)
        return state, random_move


def random_walk(initial_state, max_restarts=20, max_moves=200, verbose=False, test_mode=False):
    """
    Performs the random walk algorithm on the given initial state and returns a solution if found. Restarts if stuck.

    :param initial_state: the initial state
    :param max_restarts: the maximum number of restarts
    :param max_moves: the maximum number of moves (solution length)
    :param verbose: if true, does some printing; for debugging purposes
    :param test_mode: if true, returns additional information; for testing purposes
    :return: the solution (list of moves)
    """
    solution = []
    restarts = 0
    state = initial_state

    while not state.is_goal():
        state, move = do_random_move(state, verbose=verbose)
        if not state or not move:
            restarts += 1
            solution = []
            state = initial_state

        if move:
            solution.append(move)

        if len(solution) > max_moves:
            restarts += 1
            solution = []
            state = initial_state

        if verbose:
            print(f"{restarts}:{len(solution)}")

        if restarts > max_restarts:
            if test_mode:
                return None, restarts
            return None

    if test_mode:
        return solution, restarts
    return solution


def heuristic1(state):
    """
    Heuristic 1, as described in the paper.

    :param state: the state to compute the heuristic on
    :return: the heuristic value
    """
    return len(state.bins) - sum(b.is_goal_bin() for b in state.bins)


def heuristic2(state):
    """
    Heuristic 2, as described in the paper.

    :param state: the state to compute the heuristic on
    :return: the heuristic value
    """
    return sum(max(len(b.water) - 1, 0) for b in state.bins)


def heuristic3(state):
    """
    Heuristic 3, as described in the paper.
    :param state: the state to compute the heuristic on
    :return: the heuristic value
    """
    bottom_colors = [c.color for b in state.bins if (c := b.get_bottom())]
    return heuristic2(state) + len(bottom_colors) - len(set(bottom_colors))


def get_best_move(state, heuristic, current_value, sideways=False, verbose=False):
    """
    Returns the best move from the given state that has a heuristic value greater than (or equal to) the given value
    using the given heuristic. Used for hill climbing.

    :param state: the initial state
    :param heuristic: the heuristic function
    :param current_value: current value of the heuristic (hill climbing)
    :param sideways: if true, sideways allowed
    :param verbose: if true, prints the best move and its heuristic value; for debugging purposes
    :return: the best move if there is one
    """
    # To store the best move info
    best_move = None
    best_heuristic = None

    # Get all legal moves
    moves = state.get_legal_moves()

    # For each move
    for move in moves:
        # Generate a new state by applying the move
        new_state = state.move(move, verbose=verbose, inplace=False)

        # Compute the heuristic value
        heur_val = heuristic(new_state)

        # If sideways allowed consider the equality case
        # Find best move/heuristic
        if sideways:
            if heur_val <= current_value and (not best_move or heur_val < best_heuristic):
                best_move = move
                best_heuristic = heur_val
        else:
            if heur_val < current_value and (not best_move or heur_val < best_heuristic):
                best_move = move
                best_heuristic = heur_val

    if verbose:
        print(best_move, best_heuristic)

    return best_move

def hill_climbing(initial_state, heuristic, sideways=False, max_moves=100, verbose=False):
    """
    Applies the hill climbing algorithm on the given initial state using the given heuristic function. Returns a
    solution, if found (here we use heuristic instead of evaluation function).

    :param initial_state: the initial state
    :param heuristic: the heuristic function
    :param sideways: if true, sideways moves allowed
    :param max_moves: the maximum number of moves
    :param verbose: if true, prints some info; for debugging purposes
    :return: the solution (list of moves)
    """

    # To store the solution
    solution = []

    # Compute the heuristic of the initial state
    state = initial_state
    heuristic_value = heuristic(state)

    # While a goal state is not found
    while not state.is_goal():

        # Get the best move
        best_move = get_best_move(state, heuristic, current_value=heuristic_value, sideways=sideways, verbose=verbose)

        # If no best move (that decreases the heuristic), return None
        if not best_move:
            return None

        # Generate the new state by applying the best move and update the heuristic value
        state = state.move(best_move, verbose=verbose, inplace=False)
        heuristic_value = heuristic(state)

        # Append the best move to the solution
        solution.append(best_move)

        # If maximum solution length exceeded, return None
        if len(solution) > max_moves:
            return None

    return solution


def DFS(initial_state, max_depth=100, max_expanded_nodes=10000, graph=False, verbose=False, test_mode=False):
    """
    Applies the DFS algorithm to the given initial state. Returns a solution, if found.

    :param initial_state: the initial state
    :param max_depth: the maximum depth
    :param max_expanded_nodes: the maximum number of expanded nodes
    :param graph: if true, apply graph version of the algorithm
    :param verbose: if true, prints some info; for debugging purposes
    :param test_mode: if true, returns additional information; for testing purposes
    :return: the solution (list of moves)
    """

    # Initialize the frontier and a counter for the expanded nodes
    frontier = deque([(initial_state, [])]) # frontier stores the state and the path to that state
    num_expanded_nodes = 0

    # If graph version, initialize a set for visited states
    if graph:
        visited_states = set()

    # While there are states in the frontier
    while frontier:
        # Get the current state and path to that state
        current_state, current_path = frontier.pop()

        # If the state is goal, return the path
        if current_state.is_goal():
            if test_mode:
                return current_path, num_expanded_nodes
            return current_path

        # If graph search, add the state to visited states
        if graph:
            visited_states.add(current_state)

        # If maximum depth not exceeded
        if len(current_path) < max_depth:
            # Get all legal moves
            legal_moves = current_state.get_legal_moves()

            # For each move, generate the states and add to the frontier
            for move in legal_moves:
                new_state = current_state.move(move, verbose=verbose, inplace=False)
                num_expanded_nodes += 1

                if verbose:
                    print(num_expanded_nodes)

                # If number of expanded nodes exceeded, return None
                if num_expanded_nodes >= max_expanded_nodes:
                    if test_mode:
                        return None, num_expanded_nodes
                    return None

                # If graph version, check the visited states
                # Add the new states (with their paths) to the frontier
                if not graph or (graph and new_state not in visited_states):
                    new_path = current_path + [move]
                    frontier.append((new_state, new_path))

    if test_mode:
        return None, num_expanded_nodes
    return None


def BFS(initial_state, max_depth=100, max_expanded_nodes=10000, graph=False, verbose=False, test_mode=False):
    """
    Applies the BFS algorithm to the given initial state. Returns a solution, if found.

    :param initial_state: the initial state
    :param max_depth: the maximum depth
    :param max_expanded_nodes: the maximum number of expanded nodes
    :param graph: if true, apply graph version of the algorithm
    :param verbose: if true, prints some info; for debugging purposes
    :param test_mode: if true, returns additional information; for testing purposes
    :return: the solution (list of moves)
    """

    # Initialize the frontier and a counter for the expanded nodes
    frontier = deque([(initial_state, [])])
    num_expanded_nodes = 0

    # Early goal test
    if initial_state.is_goal():
        if test_mode:
            return [], 0
        return []

    # If graph search, add the state to visited states
    if graph:
        visited_states = set()

    # While there are elements in the frontier
    while frontier:
        current_state, current_path = frontier.popleft()

        # If graph version, initialize a set for visited states
        if graph:
            visited_states.add(current_state)

        # If maximum depth not exceeded
        if len(current_path) < max_depth:
            # Get all legal moves
            legal_moves = current_state.get_legal_moves()

            # For each move
            for move in legal_moves:
                new_state = current_state.move(move, verbose=verbose, inplace=False)
                num_expanded_nodes += 1

                if verbose:
                    print(num_expanded_nodes)

                # If number of expanded nodes exceeded, return None
                if num_expanded_nodes >= max_expanded_nodes:
                    if test_mode:
                        return None, num_expanded_nodes
                    return None

                # Update the path
                new_path = current_path + [move]

                # Early goal test
                if new_state.is_goal():
                    if test_mode:
                        return new_path, num_expanded_nodes
                    return new_path

                # If graph version, check the visited states
                # Add the new states (with their paths) to the frontier
                if not graph or (graph and new_state not in visited_states):
                    frontier.append((new_state, new_path))
    if test_mode:
        return None, num_expanded_nodes
    return None


class Node:
    """
    A class used to represent a Node. Used for A* search algorithm.

    ...

    Attributes
    ----------
    state : int
        The game state.
    cost : int
        The path cost.
    heuristic : int
        The heuristic value.
    f : int
        Path cost + heuristic value.
    path : list of tuples
        List of moves till current node.
    """
    def __init__(self, state, cost, heuristic, path):
        """
        Initialize the node.
        :param state: the state
        :param cost: the path cost
        :param heuristic: the heuristic
        :param path: the path (list of moves)
        """
        self.state = state
        self.cost = cost
        self.heuristic = heuristic
        self.f = self.cost + self.heuristic
        self.path = path

    def __ge__(self, other):
        """
        Compares the f-values (>=).
        :param other: other node
        :return: True if this node has an f-value greater than or equal to that of the other node.
        """
        return self.f >= other.f

    def __gt__(self, other):
        """
        Compares the f-values (>).
        :param other: other node
        :return: True if this node has an f-value greater than that of the other node.
        """
        return self.f >= other.f

    def __le__(self, other):
        """
        Compares the f-values (<=).
        :param other: other node
        :return: True if this node has an f-value less that of the other node.
        """
        return other.__gt__(self)

    def __lt__(self, other):
        """
        Compares the f-values (<=).
        :param other: other node
        :return: True if this node has an f-value less than or equal to that of the other node.
        """
        return other.__ge__(self)

    def __eq__(self, other):
        """
        Compares the f-values (==).
        :param other: other node
        :return: True if this and other nodes have equal f-values.
        """
        return self.f == other.f

    def __ne__(self, other):
        """
        Compares the f-values (!=).
        :param other: other node
        :return: True if this and other nodes do not have equal f-values.
        """
        return not self.__eq__(other)


def A_star(initial_state, heuristic, max_depth=100, max_expanded_nods = 10000, graph=True, verbose=False, test_mode=False):
    """
    Applies the A* search on the given initial state using the given heuristic function
    :param initial_state: the initial state
    :param heuristic: the heuristic function
    :param max_depth: the maximum depth
    :param max_expanded_nods: the maximum number of expanded nodes
    :param graph: if true, apply graph version of the algorithm
    :param verbose: if true, prints some info; for debugging purposes
    :param test_mode: if true, returns additional information; for testing purposes
    :return: the solution, if found
    """
    # Create the initial node
    initial_node = Node(initial_state, 0, heuristic(initial_state), [])

    # Initialize the frontier
    frontier = [initial_node]

    # Counter for expanded nodes
    num_expanded_nodes = 0

    # If graph version, store a visited states set and also a list of nodes
    if graph:
        states = {initial_state: initial_node}
        visited_states = set()

    # Heapify the frontier
    heapq.heapify(frontier)

    # While there are nodes in the frontier
    while frontier:

        # Pop a node from the frontier
        node = heapq.heappop(frontier)

        # Extract information from the node
        current_state, current_path, current_cost = node.state, node.path, node.cost

        # If the current state is goal, return the path
        if current_state.is_goal():
            if verbose:
                print("Goal state reached.")
            if test_mode:
                return current_path, num_expanded_nodes
            return current_path

        # If graph version, add the state to the set of visited states
        if graph:
            visited_states.add(current_state)

        # If maximum depth not exceeded
        if len(current_path) < max_depth:
            # Get all legal moves
            legal_moves = current_state.get_legal_moves()

             # For each move
            for move in legal_moves:
                # Generate the new state by applying the move
                new_state = current_state.move(move, verbose=verbose, inplace=False)
                num_expanded_nodes += 1

                if verbose:
                    print(num_expanded_nodes)

                # If the maximum number of expanded nodes exceeded, return None
                if num_expanded_nodes >= max_expanded_nods:
                    if test_mode:
                        return None, num_expanded_nodes
                    return None

                # Update the cost and the path
                new_cost = current_cost + 1
                new_path = current_path + [move]

                # Create the new ndoe
                new_node = Node(new_state, new_cost, heuristic(new_state), new_path)

                # If tree version, push the node to the frontier
                if not graph:
                    heapq.heappush(frontier, new_node)

                # If graph version, update the node if already in visited nodes
                if graph:
                    if new_state in visited_states:
                        if states[new_state].f > new_node.f:
                            frontier.remove(states[new_state])
                    states[new_state] = new_node

                    # Push the new node to the frontier
                    heapq.heappush(frontier, new_node)

    if verbose:
        print("Solution not found.")

    if test_mode:
        return None, num_expanded_nodes
    return None


def solve(initial_state, algo_active_box, heur_active_box, graph_active_box, sideways_active_box):
    """
    Attempts to solve the puzzle (the initial state) using the given algorithm and other parameters. Heavily
    intertwined with the pygame structure.

    :param initial_state: the initial state
    :param algo_active_box: boolean mask of the algorithm
    :param heur_active_box: boolean mask of the heuristic
    :param graph_active_box: boolean mask of tree/graph
    :param sideways_active_box: boolean mask of sideways/no-sideways
    :return: the solution, if found
    """
    # If algorithm not selected, return
    if not any(algo_active_box):
        return None, "Algorithm?"

    # List of algorithms (to apply the mask)
    algorithms = [BFS, DFS, A_star, random_walk, hill_climbing]

    # Get the selected algorithm
    algo = algorithms[np.argmax(algo_active_box)]

    # If hill climbing or A* selected
    if algo == hill_climbing or algo == A_star:

        # If heuristic not selected, return
        if not any(heur_active_box):
            return None, "Heuristic?"

        # List of heuristics (to apply the mask)
        heuristics = [heuristic1, heuristic2, heuristic3]

        # Get the selected heuristic
        heuristic = heuristics[np.argmax(heur_active_box)]

        # If hill climbing selected
        if algo == hill_climbing:

            # If sideways option not selected
            if not any(sideways_active_box):
                return None, "Sideways?"

            # Get sideways parameter
            sideways = sideways_active_box[0]

            # Solve the puzzle
            solution = algo(initial_state, heuristic, sideways=sideways)
        else:
            # If tree/graph option not selected
            if not any(graph_active_box):
                return None, "Tree or Graph?"

            # Get tree/graph parameter
            graph = graph_active_box[1]

            # Solve the puzzle
            solution = algo(initial_state, heuristic, graph=graph)

    # If BFS or DFS selected
    elif algo == BFS or algo == DFS:
        # If tree/graph option not selected
        if not any(graph_active_box):
            return None, "Tree or Graph?"

        # Get tree/graph parameter
        graph = graph_active_box[1]

        # Solve the puzzle
        solution = algo(initial_state, graph=graph)
    else:
        # Solve the puzzle (no additional parameters required)
        solution = algo(initial_state)

    # Return the solution (can potentially be None; pygame checks)
    return solution, "solution"
