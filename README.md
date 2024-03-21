# Water Sort Puzzle Solver
> This project solves the infamous Water Sort Puzzle using AI search algorithms.

In this project we implemented uninformed (BFS, DFS), informed (A*), and local (Random Walk, Hill Climbing) search strategies in order to solve the Water Sort Puzzle. The project also includes a UI, where the user can generate a random puzzle by selecting the game state configurations, play the game or solve the displayed puzzle with one of the above-mentioned algorithms. Everything is implemented with Python.

![An example of water sort puzzle state.](water_sort.png)

## Usage

Use the executable file to open the UI and test the algorithms interactively.

Additionally, use a Python IDE to run the algorithms manually. Usage example:

```
from water_sort import *

# Solve a randomly generated state with the Random Walk algorithm
game_state = Game_State()
solution = random_walk(game_state)
```

The returned solution is a list of 3-tuples, which includes the following information:
- the source tube index,
- the destination tube index,
- the volume of liquid to be moved.

For example, `(2,3,1)` means move `1` unit of water from tube `2` to tube `3`.

## Available algorithms
```
# Uninformed Search
BFS(initial_state, max_depth=100, max_expanded_nodes=10000,
    graph=False, verbose=False, test_mode=False)

DFS(initial_state, max_depth=100, max_expanded_nodes=10000,
    graph=False, verbose=False, test_mode=False)


# Informed Search
A_star(initial_state, heuristic, max_depth=100,
       max_expanded_nods = 10000, graph=True, verbose=False,
       test_mode=False)


# Local Search
random_walk(initial_state, max_restarts=20, max_moves=200,
            verbose=False, test_mode=False)

hill_climbing(initial_state, heuristic, sideways=False,
              max_moves=100, verbose=False)
```

## OOP Representation of the Solver
`Water_Unit`: A class used to represent a water unit.

```
Attributes
----------
color : The liquid's color.
volume : The liquid's volume.

Methods
-------
add(vol): Add the given volume to the water unit.
take(vol): Takes the given volume from the water unit.
```

`Bin`: A class used to represent a game state.

```
Attributes
----------
water : List of Bin objects representing the tubes.

free : The tube volume.

Methods
-------
get_top(): Returns the top water unit if the bin is not empty.

get_bottom(): Returns the bottom water unit if the bin is not empty.

add(new_unit): Adds the new water unit to the bin.

take(vol): Takes the given volume of water from the top of the bin.

pop(): Removes the topmost water unit. Updates the free volume.

is_empty():Returns true if the bin is empty.

is_complete():Returns true if the bin is complete (only one color).

is_goal_bin(): Returns true if the bin is a goal bin (empty or complete).

is_full(): Returns true if the bin is full.

is_fully_complete(): Returns true if the bin is fully complete (only one color and no free space).

to_list(): Converts the bin to a list. This is needed for the pygame.
```

`Game_State`: A class used to represent a game state.

```
Attributes
----------
bins : List of Bin objects representing the tubes.

bin_volume : The tube volume.

filled_bins : Number of filled bins.

empty_bins : Number of empty bins.

colors : Number of colors.

Methods
-------
get_legal_moves(): Finds and returns a list of legal moves for this state (in increasing order).

any_legal_moves(src, dest): Returns the move, if there is one from a given tube to another given tube, None otherwise.

move(move, verbose=False, inplace=True): Makes the move and returns the new state, or updates this state.

is_goal(): Checks if state is goal and returns true if it is so.

is_great_goal(self): Checks if state is goal and all non-empty tubes are full. Returns true if it is so.

to_list(): Converts the state to a list. This is needed for the pygame.
```


## About Us

We are Team Oregon: Ani Aloyan, Alexander Shahramanyan, Elina Ohanjanyan, Seda Bayadyan.