import csv
import itertools
import time
from typing import List, Tuple, Iterable, Set
import os
from multiprocessing import Pool, cpu_count

#========================================Board Basics========================================
N = 5
SIZE = N * N
ALL_CELLS = tuple(range(SIZE))
OUTPUT_FILE_NAME = "tic_tac_toe_layer"
OUTPUT_FILE_TYPE = ".csv"



base_marks_3x3 = [
    # X Cases
    (1, 1, 1,
     0, 0, 0,
     0, 0, 0),

    (0, 0, 0,
     1, 1, 1,
     0, 0, 0),

    (1, 0, 0,
     0, 1, 0,
     0, 0, 1),

    # O Cases 
    (2, 2, 2,
     0, 0, 0,
     0, 0, 0),

    (0, 0, 0,
     2, 2, 2,
     0, 0, 0),

    (2, 0, 0,
     0, 2, 0,
     0, 0, 2),
]



base_marks_4x4 = [
    # X Cases
    (1, 1, 1, 1,
     0, 0, 0, 0,
     0, 0, 0, 0,
     0, 0, 0, 0),
 
    (0, 0, 0, 0,
     1, 1, 1, 1,
     0, 0, 0, 0,
     0, 0, 0, 0),

    (1, 0, 0, 0,
     0, 1, 0, 0,
     0, 0, 1, 0,
     0, 0, 0, 1),

    # O Cases
    (2, 2, 2, 2,
     0, 0, 0, 0,
     0, 0, 0, 0,
     0, 0, 0, 0),

    (0, 0, 0, 0,
     2, 2, 2, 2,
     0, 0, 0, 0,
     0, 0, 0, 0),

    (2, 0, 0, 0,
     0, 2, 0, 0,
     0, 0, 2, 0,
     0, 0, 0, 2),
]



base_marks_5x5 = [
    # X Cases
    (1, 1, 1, 1, 1,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0),
    
    (0, 0, 0, 0, 0,
     1, 1, 1, 1, 1,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0),

    (0, 0, 0, 0, 0,
     0, 0, 0, 0, 0,
     1, 1, 1, 1, 1,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0),

    (1, 0, 0, 0, 0,
     0, 1, 0, 0, 0,
     0, 0, 1, 0, 0,
     0, 0, 0, 1, 0,
     0, 0, 0, 0, 1),

    # O Cases
    (2, 2, 2, 2, 2,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0),

    (0, 0, 0, 0, 0,
     2, 2, 2, 2, 2,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0),
    
    (0, 0, 0, 0, 0,
     0, 0, 0, 0, 0,
     2, 2, 2, 2, 2,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0),

    (2, 0, 0, 0, 0,
     0, 2, 0, 0, 0,
     0, 0, 2, 0, 0,
     0, 0, 0, 2, 0,
     0, 0, 0, 0, 2),
]

def rc_to_i(r:int, c:int) -> int:
    return r * N + c



def i_to_rc(i:int) -> Tuple[int,int]:
    return divmod(i, N)



#================================D4 Symmetries (8 transform)=================================
def t_identity(r,c):      return (r, c)
def t_rot90(r,c):         return (c, N-1-r)
def t_rot180(r,c):        return (N-1-r, N-1-c)
def t_rot270(r,c):        return (N-1-c, r)
def t_reflect_h(r,c):     return (N-1-r, c)
def t_reflect_v(r,c):     return (r, N-1-c)
def t_reflect_main(r,c):  return (c, r)
def t_reflect_anti(r,c):  return (N-1-c, N-1-r)



TRANSFORMS = [
    t_identity, t_rot90, t_rot180, t_rot270,
    t_reflect_h, t_reflect_v, t_reflect_main, t_reflect_anti
]


def transform_board(board: List[int], T) -> Tuple[int, ...]:
    out = [0] * SIZE
    for i, v in enumerate(board):
        if v == 0:
            continue
        r, c = i_to_rc(i)
        r2, c2 = T(r, c)
        out[rc_to_i(r2, c2)] = v
    return tuple(out)



def canonical_board(board: List[int]) -> Tuple[int, ...]:
    variants = [transform_board(board, T) for T in TRANSFORMS]
    return min(variants)




#======================================Generate Dataset======================================
def generate_dataset_by_layer(base_cases: List[Tuple[int]], layer: int):
    seen = set()
    folder_path = f"./data_{N}x{N}/"
    os.makedirs(folder_path, exist_ok=True)  # ✅ tự tạo thư mục nếu chưa có

    file_name = OUTPUT_FILE_NAME + f"_{layer}" + OUTPUT_FILE_TYPE
    full_path = os.path.join(folder_path, file_name)

    with open(full_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        if (layer % 2) == 1:
            # Layer lẻ -> lượt X -> chỉ sinh từ các base thắng của X (nửa đầu)
            selected_bases = base_cases[:len(base_cases)//2]
        else:
            # Layer chẵn -> lượt O -> chỉ sinh từ các base thắng của O (nửa sau)
            selected_bases = base_cases[len(base_cases)//2:]

        for base in selected_bases:
            generate_dataset_by_base_case(base, layer, writer, seen)



def generate_dataset_by_base_case(base_case: Tuple[int], current_layer: int, writer: csv.writer, seen: set):
    x_left = 0
    o_left = 0
    if (current_layer % 2) == 1: # X's turn
        x_left = current_layer // 2 + 1 - sum(1 for v in base_case if v == 1)
        o_left = current_layer // 2 - sum(1 for v in base_case if v == 2)
    else: # O's turn
        x_left = current_layer // 2 - sum(1 for v in base_case if v == 1)
        o_left = current_layer // 2 - sum(1 for v in base_case if v == 2)
    
    empty_cells = [i for i, v in enumerate(base_case) if v == 0]
    for cells in itertools.combinations(empty_cells, x_left + o_left):
        for x_cells in itertools.combinations(cells, x_left):
            o_cells = set(cells) - set(x_cells)
            new_case = list(base_case)
            for x in x_cells:
                new_case[x] = 1
            for o in o_cells:
                new_case[o] = 2
            if not valid_case(new_case): continue
            cb = canonical_board(new_case)
            key = tuple(cb)

            if key not in seen:
                seen.add(key)
                writer.writerow(cb)



#=======================================Validate Case========================================
def valid_case(board: List[int]) -> bool:
    """
    Check if the board state is valid.
    - Both players cannot win simultaneously
    - At most 2 winning lines allowed (must be vertical×horizontal or vertical/horizontal×diagonal)
    - Special case: 5x5 can have 3 lines (middle row × middle column × main diagonal)
    """
    x_lines = count_winning_lines(board, 1)
    o_lines = count_winning_lines(board, 2)
    
    # Both players cannot win simultaneously
    if x_lines > 0 and o_lines > 0:
        return False
    
    # Check for the special 5x5 case with 3 winning lines
    if N == 5 and (x_lines == 3 or o_lines == 3):
        return is_special_5x5_case(board, 1 if x_lines == 3 else 2)
    
    # Maximum 2 winning lines for general cases
    if x_lines > 2 or o_lines > 2:
        return False
    
    # If exactly 2 winning lines, verify they intersect correctly
    if x_lines == 2:
        if not is_valid_double_win(board, 1):
            return False
    
    if o_lines == 2:
        if not is_valid_double_win(board, 2):
            return False
    
    return True



def count_winning_lines(board: List[int], player: int) -> int:
    """
    Count winning lines for a player.
    """
    count = 0
    
    # Check rows
    for r in range(N):
        if all(board[r * N + c] == player for c in range(N)):
            count += 1
    
    # Check columns
    for c in range(N):
        if all(board[r * N + c] == player for r in range(N)):
            count += 1
    
    # Check main diagonal
    if all(board[i * N + i] == player for i in range(N)):
        count += 1
    
    # Check anti-diagonal
    if all(board[i * N + (N - 1 - i)] == player for i in range(N)):
        count += 1
    
    return count



def get_winning_line_types(board: List[int], player: int) -> List[str]:
    """
    Get types of winning lines: 'row', 'col', 'main_diag', 'anti_diag'
    """
    types = []
    
    # Check rows
    for r in range(N):
        if all(board[r * N + c] == player for c in range(N)):
            types.append('row')
    
    # Check columns
    for c in range(N):
        if all(board[r * N + c] == player for r in range(N)):
            types.append('col')
    
    # Check main diagonal
    if all(board[i * N + i] == player for i in range(N)):
        types.append('main_diag')
    
    # Check anti-diagonal
    if all(board[i * N + (N - 1 - i)] == player for i in range(N)):
        types.append('anti_diag')
    
    return types



def is_valid_double_win(board: List[int], player: int) -> bool:
    """
    Check if 2 winning lines are valid intersections:
    - row × col (vertical × horizontal)
    - row × diagonal (horizontal × diagonal)
    - col × diagonal (vertical × diagonal)
    """
    types = get_winning_line_types(board, player)
    
    if len(types) != 2:
        return False
    
    type_set = set(types)
    
    # Valid combinations
    if (N == 4):
        valid_combinations = [
            {'row', 'col'},           # vertical × horizontal
            {'row', 'main_diag'},     # horizontal × diagonal
            {'row', 'anti_diag'},     # horizontal × diagonal
            {'col', 'main_diag'},     # vertical × diagonal
            {'col', 'anti_diag'},     # vertical × diagonal
        ]
    else:  # N == 3 or N == 5
        valid_combinations = [
            {'row', 'col'},           # vertical × horizontal
            {'row', 'main_diag'},     # horizontal × diagonal
            {'row', 'anti_diag'},     # horizontal × diagonal
            {'col', 'main_diag'},     # vertical × diagonal
            {'col', 'anti_diag'},     # vertical × diagonal
            {'main_diag', 'anti_diag'} # diagonal × diagonal
        ]
    
    return type_set in valid_combinations



def is_special_5x5_case(board: List[int], player: int) -> bool:
    """
    Check if it's the special 5x5 case: middle row, middle column, and main diagonal.
    """
    types = get_winning_line_types(board, player)
    
    if len(types) != 3:
        return False
    
    # Must be exactly row, col, and main_diag
    type_set = set(types)
    return type_set == {'row', 'col', 'main_diag'}



#============================================Main============================================
def generate_dataset():
    base_cases = []
    start_layer = 0
    print(f"🔹 Generating dataset for {N}x{N} from layer {start_layer} to {SIZE}...")
    total_start = time.time()

    if N == 3:
        base_cases = base_marks_3x3
        start_layer = 5
    elif N == 4:
        base_cases = base_marks_4x4
        start_layer = 7
    elif N == 5:
        base_cases = base_marks_5x5
        start_layer = 9
    
    for layer in range(start_layer, SIZE + 1):
        start = time.time()
        generate_dataset_by_layer(base_cases, layer)
        end = time.time()
        print(f"✅ Layer {layer} done in {end - start:.2f} seconds")
    
    total_end = time.time()
    print(f"\n🏁 All layers completed in {total_end - total_start:.2f} seconds\n")
    
if __name__ == "__main__":
    generate_dataset()