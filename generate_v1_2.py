import itertools
import time
import struct
from typing import List, Tuple, Iterable, Set

# ----- Board basics -----
N = 4
SIZE = N * N
ALL_CELLS = tuple(range(SIZE))

def rc_to_i(r:int, c:int) -> int:
    return r * N + c

def i_to_rc(i:int) -> Tuple[int,int]:
    return divmod(i, N)

# ----- D4 symmetries (8 transforms) -----
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
    out = [0]*SIZE
    for i,v in enumerate(board):
        if v == 0: 
            continue
        r,c = i_to_rc(i)
        r2,c2 = T(r,c)
        out[rc_to_i(r2,c2)] = v
    return tuple(out)

def canonical_board(board: List[int]) -> Tuple[int, ...]:
    """Return the lexicographically smallest board among all 8 symmetries."""
    variants = [transform_board(board, T) for T in TRANSFORMS]
    return min(variants)

# ----- Read base masks -----
def read_base_masks(path: str) -> List[List[int]]:
    rows = []
    with open(path, newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")] if "," in line else line.split()
            vals = [int(p) for p in parts if p != ""]
            if len(vals) != SIZE:
                raise ValueError(f"Each row must have {SIZE} ints, got {len(vals)}: {line}")
            rows.append(vals)
    return rows

def line_positions_from_mask(mask: List[int]) -> Tuple[int,...]:
    idxs = [i for i,v in enumerate(mask) if v != 0]
    if len(idxs) != 4:
        raise ValueError("Each base mask must have exactly 4 non-zero cells.")
    vals = {mask[i] for i in idxs}
    if vals not in ({1}, {2}):
        raise ValueError("A base mask must contain either all 1s or all 2s (not mixed).")
    return tuple(sorted(idxs))

def apply_transform_to_indices(idxs: Tuple[int,...], T) -> Tuple[int,...]:
    out = []
    for i in idxs:
        r,c = i_to_rc(i)
        r2,c2 = T(r,c)
        out.append(rc_to_i(r2,c2))
    return tuple(sorted(out))

def expand_lines_via_symmetry(base_masks: List[List[int]]) -> List[Tuple[int,...]]:
    """Unique geometric winning lines from base masks using all symmetries."""
    lines = set()
    for m in base_masks:
        base_idxs = line_positions_from_mask(m)
        for T in TRANSFORMS:
            lines.add(apply_transform_to_indices(base_idxs, T))
    return sorted(lines)  # expect 10 lines on 4x4

# ----- Win/validation helpers -----
def has_winner(board: List[int], player: int, win_lines: List[Tuple[int,...]]) -> bool:
    return any(all(board[i] == player for i in line) for line in win_lines)

def winners(board: List[int], win_lines: List[Tuple[int,...]]) -> Set[int]:
    s = set()
    if has_winner(board, 1, win_lines): s.add(1)
    if has_winner(board, 2, win_lines): s.add(2)
    return s

def count_pieces(board: List[int]) -> Tuple[int, int]:
    x = sum(1 for v in board if v == 1)
    o = sum(1 for v in board if v == 2)
    return x, o

def ply(board: List[int]) -> int:
    return sum(1 for v in board if v != 0)

def valid_turn_counts(board: List[int], win_lines: List[Tuple[int,...]]) -> bool:
    x, o = count_pieces(board)
    if not (x == o or x == o + 1):                 # X first
        return False
    if has_winner(board, 2, win_lines) and x != o:     # O wins → even ply
        return False
    if has_winner(board, 1, win_lines) and x != o + 1: # X wins → odd ply
        return False
    return True

def min_ply_ok(board: List[int], win_lines: List[Tuple[int,...]]) -> bool:
    x, o = count_pieces(board)
    if has_winner(board, 1, win_lines) and (2 * x - 1) < 7:  # X's 4th move
        return False
    if has_winner(board, 2, win_lines) and (2 * o) < 8:      # O's 4th move
        return False
    return True

def is_terminal_reachable(board: List[int], winner: int, win_lines: List[Tuple[int,...]]) -> bool:
    """
    Exactly one winner. There exists a last-move on a winning line such that
    removing it yields a legal previous position: correct counts, no winner, turn rule holds.
    """
    if winner not in (1,2): return False
    if winners(board, win_lines) != {winner}: return False

    x, o = count_pieces(board)
    total = x + o

    if winner == 1:
        if total % 2 != 1 or x != o + 1: return False
        prev_x, prev_o = x - 1, o
    else:
        if total % 2 != 0 or x != o: return False
        prev_x, prev_o = x, o - 1

    for line in win_lines:
        if all(board[i] == winner for i in line):
            for i in line:
                if board[i] != winner: 
                    continue
                prev = board.copy()
                prev[i] = 0
                if count_pieces(prev) != (prev_x, prev_o): 
                    continue
                if winners(prev, win_lines): 
                    continue
                if not (prev_x == prev_o or prev_x == prev_o + 1):
                    continue
                return True
    return False

# ----- Exhaustive (deterministic) enumeration -----
def enumerate_wins_for_line(line: Tuple[int,...], winner: int, win_lines: List[Tuple[int,...]]) -> Iterable[List[int]]:
    """All terminal, reachable boards where 'winner' wins on 'line'."""
    if winner == 1:
        # Odd plies: 7..15
        ply_values = range(7, 16, 2)
    else:
        # Even plies: 8..16
        ply_values = range(8, 17, 2)

    for T in ply_values:
        if winner == 1:
            x_total = (T + 1) // 2
            o_total = (T - 1) // 2
            extra_w = x_total - 4
            extra_other = o_total
            w_val, other_val = 1, 2
        else:
            x_total = T // 2
            o_total = T // 2
            extra_w = o_total - 4
            extra_other = x_total
            w_val, other_val = 2, 1

        if extra_w < 0 or extra_other < 0:
            continue

        remaining = tuple(i for i in ALL_CELLS if i not in line)
        for extra_w_cells in itertools.combinations(remaining, extra_w):
            remaining2 = tuple(i for i in remaining if i not in extra_w_cells)
            if extra_other > len(remaining2):
                continue
            for other_cells in itertools.combinations(remaining2, extra_other):
                b = [0]*SIZE
                for i in line: b[i] = w_val
                for i in extra_w_cells: b[i] = w_val
                if any(b[i] != 0 for i in other_cells):
                    continue
                for i in other_cells: b[i] = other_val

                if winners(b, win_lines) != {winner}:
                    continue
                if not valid_turn_counts(b, win_lines):
                    continue
                if not min_ply_ok(b, win_lines):
                    continue
                if not is_terminal_reachable(b, winner, win_lines):
                    continue

                yield b

# ----- Binary encoding/decoding -----
def board_to_bits(board: List[int]) -> int:
    """
    Encode board as 32-bit integer using 2 bits per cell:
    00 = empty, 01 = X (player 1), 10 = O (player 2)
    """
    result = 0
    for i, val in enumerate(board):
        result |= (val << (i * 2))
    return result

def bits_to_board(bits: int) -> List[int]:
    """Decode 32-bit integer back to board."""
    board = []
    for i in range(SIZE):
        board.append((bits >> (i * 2)) & 3)
    return board

def encode_record(board: List[int], winner: int, ply_count: int) -> bytes:
    """
    Encode a single record as 5 bytes:
    - 4 bytes: board (32 bits, 2 bits per cell)
    - 1 byte: metadata (4 bits winner, 4 bits ply)
    """
    board_bits = board_to_bits(board)
    metadata = ((winner & 0xF) << 4) | (ply_count & 0xF)
    return struct.pack('<IB', board_bits, metadata)

def decode_record(data: bytes) -> Tuple[List[int], int, int]:
    """Decode 5-byte record back to board, winner, ply."""
    board_bits, metadata = struct.unpack('<IB', data)
    board = bits_to_board(board_bits)
    winner = (metadata >> 4) & 0xF
    ply_count = metadata & 0xF
    return board, winner, ply_count

# ----- I/O -----
def write_binary_dataset(path: str, records: List[Tuple[List[int], int, int]]):
    """
    Write binary dataset with header:
    - 4 bytes: magic number (0x54545434 = "TTT4")
    - 4 bytes: record count
    - 4 bytes: record size (always 5)
    - N * 5 bytes: records
    """
    MAGIC = 0x54545434  # "TTT4"
    RECORD_SIZE = 5
    
    with open(path, 'wb') as f:
        # Write header
        f.write(struct.pack('<III', MAGIC, len(records), RECORD_SIZE))
        
        # Write records
        for board, winner, ply_count in records:
            f.write(encode_record(board, winner, ply_count))

def read_binary_dataset(path: str) -> List[Tuple[List[int], int, int]]:
    """Read binary dataset."""
    with open(path, 'rb') as f:
        # Read header
        magic, count, record_size = struct.unpack('<III', f.read(12))
        if magic != 0x54545434:
            raise ValueError(f"Invalid magic number: {hex(magic)}")
        if record_size != 5:
            raise ValueError(f"Unexpected record size: {record_size}")
        
        # Read records
        records = []
        for _ in range(count):
            data = f.read(record_size)
            records.append(decode_record(data))
        
        return records

def format_size(bytes_count: int) -> str:
    """Format byte count into readable string."""
    if bytes_count < 1024:
        return f"{bytes_count} bytes"
    elif bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.2f} KB"
    else:
        return f"{bytes_count / (1024 * 1024):.2f} MB"

def format_time(seconds: float) -> str:
    """Format seconds into readable time string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.2f}s"

# ----- Main (symmetric-only) with timing -----
def generate_dataset_symmetric_only(
    base_path: str = "base_dataset.csv",
    out_path: str = "tictactoe4x4_dataset.bin"
) -> None:
    overall_start = time.time()
    
    print("="*60)
    print("4x4 TIC-TAC-TOE DATASET GENERATOR (BINARY FORMAT)")
    print("="*60)
    
    # Read base masks
    print(f"\n[1/3] Reading base masks from '{base_path}'...")
    step_start = time.time()
    base_masks = read_base_masks(base_path)
    step_time = time.time() - step_start
    print(f"  ✓ Loaded {len(base_masks)} base masks in {format_time(step_time)}")
    
    # Expand lines via symmetry
    print(f"\n[2/3] Expanding winning lines via symmetry...")
    step_start = time.time()
    win_lines = expand_lines_via_symmetry(base_masks)
    step_time = time.time() - step_start
    print(f"  ✓ Generated {len(win_lines)} unique winning lines in {format_time(step_time)}")
    
    # Generate all boards
    print(f"\n[3/3] Generating terminal boards...")
    step_start = time.time()
    
    seen = set()
    records: List[Tuple[List[int], int, int]] = []
    
    for idx, line in enumerate(win_lines, 1):
        line_start = time.time()
        line_boards = 0
        
        for winner in (1, 2):
            for b in enumerate_wins_for_line(line, winner, win_lines):
                cb = canonical_board(b)
                key = (cb, winner)
                if key in seen:
                    continue
                seen.add(key)
                records.append((list(cb), winner, ply(b)))
                line_boards += 1
        
        line_time = time.time() - line_start
        elapsed = time.time() - step_start
        progress = (idx / len(win_lines)) * 100
        
        print(f"  Line {idx}/{len(win_lines)} [{progress:5.1f}%]: "
              f"+{line_boards} boards ({len(records)} total) | "
              f"{format_time(line_time)} | "
              f"Elapsed: {format_time(elapsed)}")
    
    step_time = time.time() - step_start
    print(f"  ✓ Generated {len(records)} unique boards in {format_time(step_time)}")
    
    # Write output
    print(f"\n[4/4] Writing binary dataset to '{out_path}'...")
    write_start = time.time()
    write_binary_dataset(out_path, records)
    write_time = time.time() - write_start
    
    import os
    file_size = os.path.getsize(out_path)
    print(f"  ✓ File written in {format_time(write_time)}")
    print(f"  ✓ File size: {format_size(file_size)}")
    
    # Summary
    total_time = time.time() - overall_start
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Total unique boards: {len(records)}")
    print(f"File size: {format_size(file_size)}")
    print(f"Bytes per record: {file_size / len(records):.2f}")
    print(f"Total time: {format_time(total_time)}")
    print(f"Average time per board: {(total_time/len(records)*1000):.2f}ms")
    print(f"\nBinary format: 5 bytes per record")
    print(f"  - 4 bytes: board state (2 bits per cell)")
    print(f"  - 1 byte: winner (4 bits) + ply (4 bits)")
    print("="*60 + "\n")

# ----- Demo reading -----
def demo_read(path: str = "tictactoe4x4_dataset.bin", num_samples: int = 5):
    """Demonstrate reading the binary dataset."""
    print(f"\n{'='*60}")
    print("READING BINARY DATASET")
    print('='*60)
    
    records = read_binary_dataset(path)
    print(f"Loaded {len(records)} records\n")
    
    print(f"First {num_samples} records:")
    for i, (board, winner, ply_count) in enumerate(records[:num_samples]):
        print(f"\nRecord {i+1}:")
        for row in range(N):
            line = ""
            for col in range(N):
                val = board[row * N + col]
                line += ['-', 'X', 'O'][val] + " "
            print(f"  {line}")
        print(f"  Winner: {'X' if winner == 1 else 'O'}, Ply: {ply_count}")
    
    print('='*60)

if __name__ == "__main__":
    generate_dataset_symmetric_only(
        base_path="base_dataset.csv",
        out_path="tictactoe4x4_dataset.bin"
    )
    
    # Optionally demo reading
    # demo_read("tictactoe4x4_dataset.bin")