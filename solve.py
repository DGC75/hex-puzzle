import argparse
import random
import time
from typing import List, Optional, Tuple
import copy

import yaml

import matplotlib.pyplot as plt
from matplotlib.widgets import Button


from grid import (
    allowed_xs_list,
    allowed_ys_lists,
    BLOCKED,
    Grid,
)

from pieces import get_piece, NUM_PIECES, Piece, rot_list

# Global variables for slideshow
fig = None
ax = None
grid_states = []  # List to store all grid states
current_state_idx = -1  # Index of the currently displayed state
auto_play = False  # Flag to control automatic playback

def setup_slideshow():
    """Initialize the figure, axis and controls for the interactive slideshow."""
    global fig, ax, prev_button_ax, next_button_ax, play_button_ax, prev_button, next_button, play_button
    
    # Create figure with space for buttons at the bottom
    fig = plt.figure(figsize=(8, 7))
    
    # Main plot area for the grid
    ax = plt.axes([0.1, 0.2, 0.8, 0.7])  # [left, bottom, width, height]
    ax.set(xlim=(2, 23), ylim=(-3, 18))
    ax.set_aspect("equal")
    plt.axis("on")
    
    # Button for previous state
    prev_button_ax = plt.axes([0.2, 0.05, 0.15, 0.06])
    prev_button = Button(prev_button_ax, 'Previous')
    prev_button.on_clicked(on_prev_clicked)
    
    # Button for play/pause
    play_button_ax = plt.axes([0.4, 0.05, 0.15, 0.06])
    play_button = Button(play_button_ax, 'Play')
    play_button.on_clicked(on_play_clicked)
    
    # Button for next state
    next_button_ax = plt.axes([0.6, 0.05, 0.15, 0.06])
    next_button = Button(next_button_ax, 'Next')
    next_button.on_clicked(on_next_clicked)
    
    # Text for state counter
    counter_ax = plt.axes([0.8, 0.05, 0.15, 0.06])
    counter_ax.axis('off')  # Hide axis
    
    plt.ion()  # Turn on interactive mode
    plt.show()
    return fig, ax

def on_prev_clicked(event):
    """Handle click on previous button."""
    global current_state_idx
    if current_state_idx > 0:
        current_state_idx -= 1
        update_display()

def on_next_clicked(event):
    """Handle click on next button."""
    global current_state_idx
    if current_state_idx < len(grid_states) - 1:
        current_state_idx += 1
        update_display()

def on_play_clicked(event):
    """Toggle play/pause of the slideshow."""
    global auto_play
    auto_play = not auto_play
    play_button.label.set_text('Pause' if auto_play else 'Play')
    if auto_play:
        play_slideshow()

def play_slideshow():
    """Automatically play the slideshow."""
    global current_state_idx, auto_play
    while auto_play and current_state_idx < len(grid_states) - 1:
        current_state_idx += 1
        update_display()
        plt.pause(0.3)  # Adjust speed as desired
    
    if current_state_idx >= len(grid_states) - 1:
        auto_play = False
        play_button.label.set_text('Play')

def update_display():
    """Update the display with the current grid state."""
    global fig, ax, current_state_idx
    
    # Clear previous grid visualization
    ax.clear()
    
    # Draw the current grid state
    grid_states[current_state_idx].draw(ax=ax)
    ax.set(xlim=(2, 23), ylim=(-3, 18))
    ax.set_aspect("equal")
    plt.axis("on")
    
    # Update state counter text
    ax.set_title(f"State {current_state_idx+1}/{len(grid_states)}")
    
    # Update the display
    fig.canvas.draw()
    fig.canvas.flush_events()

def show_solution(grid: Grid):
    """
    Update the slideshow with the current grid state.
    
    This function stores each grid state and updates the display.
    """
    global fig, ax, grid_states, current_state_idx
    
    # Initialize figure and axis if they don't exist
    if fig is None or ax is None:
        fig, ax = setup_slideshow()
    
    # Store a deep copy of the current grid state
    grid_copy = Grid()
    grid_copy.grid = grid.grid.copy()
    grid_states.append(grid_copy)
    current_state_idx = len(grid_states) - 1
    
    # Update the display
    update_display()
    
    # Small pause to avoid overwhelming the system
    plt.pause(0.05)

def solve_recursive(
    grid: Grid,
    pieces: List[Piece],
    index: int = 0,
    check_at: int = 0,
) -> bool:
    """
    Recursive function to solve a problem.

    The idea is that, once a piece is positioned, the problem becomes an 
    easier problem, with one piece left and a different starting grid.

    This function, given a grid state (with possibly some piece already 
    positioned) and a piece (as an index in a list of pieces), tries to find a
    valid position for the piece. Once this is found, it recusively call itself.

    In case the problem is solved, the grid will contain the configuration of 
    pieces (i.e. each cell will contain the index of the piece occupying it), 
    while `pieces` will contain the concrete pieces that solve the problem.

    Args:
        grid (Grid): Problem grid.
        pieces (list): List of pieces for the problem.
        index (int): Index (in the list) of the current piece.
        check_at (int): Index from which checking if the grid is solvable 
            See grid.is_impossible() for details.
    
    Returns:
        True if the problem can be solved. False otherwise.
    """
    if index == len(pieces):
        # No more pieces to position => Solved!
        return True
    
    if index >= check_at and grid.is_impossible():
        #show_solution(grid)
        return False

    
    for x in allowed_xs_list:
        for y in allowed_ys_lists[x-1]:
            for rot in rot_list:
                piece = pieces[index].make_new(x, y, rot)

                if grid.add_piece(piece):
                    #show_solution(grid)
                    if solve_recursive(grid, pieces, index + 1, check_at):
                        pieces[index] = piece
                        return True
                    
                    grid.remove_piece(piece)
                    

    return False


# === Iterative solver ===
# Initial tests didn't show much advantage in avoiding recursion.
# Hence, this might not work now.

def config_gen(piece):
    for rot in rot_list:
        for x in allowed_xs_list:
            for y in allowed_ys_lists[x-1]:
                yield piece.make_new(x, y, rot)

def search_piece_position(grid, generator):
    for piece in generator:
        if grid.add_piece(piece):
            return piece, generator
    return None


def solve_iter(grid, pieces, check_at=5) -> bool:
    generators = [config_gen(piece) for piece in pieces]
    idx = 0

    while idx < len(pieces):
        piece, gen = pieces[idx], generators[idx]
        res = search_piece_position(grid, gen)

        if res is not None:
            # If a position is found
            if idx >= check_at and grid.check_isolated():
                grid.remove_piece(res[0])
                continue

            pieces[idx] = res[0]
            generators[idx] = res[1]
            idx += 1

        else:
            # If no position is found
            if idx == 0:
                return False

            generators[idx] = config_gen(piece)
            idx -= 1
            grid.remove_piece(pieces[idx])

    return True

# === Iterative solver ===



def prepare_problem(filename: str) -> Tuple[Grid, List[Piece]]:
    """
    Loads a problem from a YAML configuration file.

    The config file should contain the following entries:

    - 'blocked_grid_cells': This should be a list of (x, y) couples 
      corresponding to the x and y coordinates of the blocked grid cells.

    - 'excluded_pieces': This should be a list of indexes corresponding to the 
      indexes of the pieces that should be excluded when solving the problem.
      Can be empty.

    Args:
        filename (str): Configuration file name (yaml).
    
    Returns:
        Grid, List: Starting grid and list of available pieces.
    """
    with open(filename, "r") as fp:
        problem_conf = yaml.safe_load(fp)
    
    grid = Grid()
    for x, y in problem_conf["blocked_grid_cells"]:
        grid.grid[y, x] = BLOCKED
    assert not grid.is_impossible()


    pieces = [
        get_piece(i)
        for i in range(1, NUM_PIECES+1)
        if i not in problem_conf.get("excluded_pieces", ())
    ]
  

    return grid, pieces


def save_solution_to_config(pieces: List[Piece], filename: str):
    """
    Save a set of pieces as solution in the configuration file. If a solution 
    already exists in the config file, this does nothing.

    Args:
        pieces (list): List of pieces (supposedly solving the problem).
        filename (str): Problem configuration file (yaml).
    """
    with open(filename, "r") as fp:
        problem_conf = yaml.safe_load(fp)

    if not "solution" in problem_conf:
        solution  = {
            piece.index: {
                "base_x": piece.base_x,
                "base_y": piece.base_y,
                "rotation": piece.rotation,
            }
            for piece in pieces
        }
        with open(filename, "a") as fp:
            yaml.safe_dump({"solution": solution}, fp)

def solve(
    filename: str,
    seed: Optional[int] = None,
    check_at: int = 1,
    save_solution: bool = True,
    use_iterative: bool = False,
):
    """
    Solves a problem.

    The problem is loaded from a configuration file containing the initial 
    grid configuration (in terms of blocked cells) and the available pieces.

    Args:
        filename (str): Problem configuration file (yaml).
        seed (int, optional): Seed for the random number generator. This 
            influences the order of the pieces. Default: None.
        check_at (int): Number of pieces placed after which starting to check 
            if the grid is solvable. Default: 3.
        save_solution (bool): Whether to save the solution in the input config 
            file (if not already present). Default: True.
        use_iterative (bool): Ignored.
    """
    global grid_states, current_state_idx, auto_play
    
    # Reset slideshow state
    grid_states = []
    current_state_idx = -1
    auto_play = False

    grid, pieces = prepare_problem(filename)
    random.seed(seed)
    #print("seed:", seed)    
    random.shuffle(pieces)

    # Add initial grid state to slideshow
    #show_solution(grid)

    # solver = solve_iter if use_iterative else solve_recursive
    solver = solve_recursive

    print("Starting to solve problem...")
    start = time.time()
    solved = solver(grid, pieces, check_at=check_at)
    end = time.time()
    print(f"Ended in: {end - start:.2f} seconds")
    if not solved:
        print("The problem could not be solved! :'(")
        # Debugging: Show the grid state and pieces
        print("Final grid state shown in slideshow")
        print("Remaining pieces:")
        for piece in pieces:
            print(piece)
    else:
        print("Problem solved! :D")
        print(f"Found solution with {len(grid_states)} steps")
        print("Use the interactive slideshow to review the solution steps")

    if solved and save_solution:
        save_solution_to_config(pieces, filename)

    # Keep the slideshow open for interactive navigation
    if plt.get_fignums():  # Check if any figures exist
        plt.ioff()  # Turn off interactive mode
        plt.show(block=True)  # Show and block until window is closed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solves a problem.")
    parser.add_argument(
        "config_file", help="Problem configuration file (YAML)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for the piece shuffle"
    )
    parser.add_argument(
        "--check-at", type=int, default=3,
        help="Index from which checking if the current grid is solvable",
    )
    parser.add_argument(
        "--no-save-solution", action="store_false", dest="save_solution",
        help="Do not save the solution in the input config file"
    )

    args = parser.parse_args()

    solve(
        filename=args.config_file,
        seed=args.seed,
        check_at=args.check_at,
        save_solution=args.save_solution,
    )