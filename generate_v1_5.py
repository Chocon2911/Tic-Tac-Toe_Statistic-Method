import itertools
import os
import struct
import time
from typing import List, Tuple, Iterable, Set

# ====== CONFIG ======
N = 4
SIZE = N * N
ALL_CELLS = tuple(range(SIZE))
RECORD_SIZE = 8  # 8 bytes = 25 cells * 2 bits (64-bit safe)
MAGIC = 0x54545435  # "TTT5"

# ====== AUTO CALCULATE MIN LAYER ======
def calculate_min_winning_layer(n):
    """
    Tính layer tối thiểu để có thể có trận thắng hợp lệ.
    
    Để thắng cần:
    - N quân liên tiếp trên 1 hàng/cột/chéo
    - Đối thủ có ít nhất (N-1) quân (để không thắng trước)
    
    Layer tối thiểu:
    - Player 1 (X) thắng: cần N quân X và (N-1) quân O
      → total = N + (N-1) = 2N - 1
    - Player 2 (O) thắng: cần N quân O và N quân X
      → total = N + N = 2N
    
    Vì X đi trước, trận thắng sớm nhất là X thắng ở lượt 2N-1
    """
    return 2 * n - 1

MIN_LAYER = calculate_min_winning_layer(N)
MAX_LAYER = SIZE

print(f"Board size: {N}x{N}")
print(f"Minimum winning layer: {MIN_LAYER}")
print(f"Maximum layer: {MAX_LAYER}")
print()

# ====== Basic Board Utils ======
def rc_to_i(r, c): return r * N + c
def i_to_rc(i): return divmod(i, N)

# D4 symmetry transforms
def t_identity(r,c): return (r,c)
def t_rot90(r,c): return (c, N-1-r)
def t_rot180(r,c): return (N-1-r, N-1-c)
def t_rot270(r,c): return (N-1-c, r)
def t_reflect_h(r,c): return (N-1-r, c)
def t_reflect_v(r,c): return (r, N-1-c)
def t_reflect_main(r,c): return (c, r)
def t_reflect_anti(r,c): return (N-1-c, N-1-r)

TRANSFORMS = [
    t_identity, t_rot90, t_rot180, t_rot270,
    t_reflect_h, t_reflect_v, t_reflect_main, t_reflect_anti
]

def transform_board(board, T):
    out = [0]*SIZE
    for i,v in enumerate(board):
        if v == 0: continue
        r,c = i_to_rc(i)
        r2,c2 = T(r,c)
        out[rc_to_i(r2,c2)] = v
    return tuple(out)

def canonical_board(board):
    return min(transform_board(board, T) for T in TRANSFORMS)

# ====== Game Logic ======
def count_pieces(board):
    x = sum(1 for v in board if v == 1)
    o = sum(1 for v in board if v == 2)
    return x, o

def ply(board): return sum(1 for v in board if v != 0)

def has_winner(board, player, win_lines):
    return any(all(board[i] == player for i in line) for line in win_lines)

def winners(board, win_lines):
    res = set()
    if has_winner(board, 1, win_lines): res.add(1)
    if has_winner(board, 2, win_lines): res.add(2)
    return res

def valid_turn_counts(board, win_lines):
    x, o = count_pieces(board)
    if not (x == o or x == o + 1):  # X starts
        return False
    if has_winner(board, 2, win_lines) and x != o:
        return False
    if has_winner(board, 1, win_lines) and x != o + 1:
        return False
    return True

def min_ply_ok(board, win_lines):
    """Kiểm tra số lượt tối thiểu để thắng hợp lý"""
    x, o = count_pieces(board)
    # X thắng: cần ít nhất 2N-1 quân (N quân X + N-1 quân O)
    if has_winner(board, 1, win_lines) and (2 * x - 1) < MIN_LAYER:
        return False
    # O thắng: cần ít nhất 2N quân (N quân X + N quân O)
    if has_winner(board, 2, win_lines) and (2 * o) < (MIN_LAYER + 1):
        return False
    return True

def is_terminal_reachable(board, winner, win_lines):
    if winner not in (1, 2):
        return False
    if winners(board, win_lines) != {winner}:
        return False
    x, o = count_pieces(board)
    if winner == 1:
        if x != o + 1:
            return False
        prev_x, prev_o = x - 1, o
    else:
        if x != o:
            return False
        prev_x, prev_o = x, o - 1

    # Tìm tất cả các line thắng của winner
    win_lines_for_player = [line for line in win_lines if all(board[i] == winner for i in line)]
    if not win_lines_for_player:
        return False

    # Nếu có hơn 1 line, kiểm tra xem chúng có giao nhau tại đúng 1 ô không
    if len(win_lines_for_player) > 1:
        # Lấy tập giao của tất cả các line
        intersect = set(win_lines_for_player[0])
        for line in win_lines_for_player[1:]:
            intersect &= set(line)
        # Giao nhau phải đúng 1 ô và ô đó thuộc winner
        if len(intersect) != 1:
            return False
        inter_cell = next(iter(intersect))
        if board[inter_cell] != winner:
            return False

    # Kiểm tra có thể đạt được từ lượt trước (bỏ đi 1 quân của winner)
    # chỉ cần 1 ô bỏ ra mà bàn trước không còn thắng nữa
    for i, v in enumerate(board):
        if v != winner:
            continue
        prev = list(board)
        prev[i] = 0
        if count_pieces(prev) != (prev_x, prev_o):
            continue
        if winners(prev, win_lines):
            continue
        if not valid_turn_counts(prev, win_lines):
            continue
        return True
    return False


# ====== Win Lines ======
def expand_lines_via_symmetry(base_masks):
    lines = set()
    for m in base_masks:
        idxs = [i for i,v in enumerate(m) if v]
        for T in TRANSFORMS:
            out = []
            for i in idxs:
                r,c = i_to_rc(i)
                r2,c2 = T(r,c)
                out.append(rc_to_i(r2,c2))
            lines.add(tuple(sorted(out)))
    return sorted(lines)

# ====== Encoding / IO ======
def board_to_bits(board):
    result = 0
    for i,val in enumerate(board):
        result |= (val << (i*2))
    return result

def encode_record(board): 
    return struct.pack('<Q', board_to_bits(board))  # use 64-bit

def init_layer_file(path):
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', MAGIC, 0))  # magic + count placeholder

def append_record(path, board):
    with open(path, 'ab') as f:
        f.write(encode_record(board))

def finalize_count(path):
    with open(path, 'rb+') as f:
        magic, count = struct.unpack('<II', f.read(8))
        actual_count = (os.path.getsize(path) - 8) // RECORD_SIZE
        f.seek(4)
        f.write(struct.pack('<I', actual_count))

# ====== Main enumeration ======
def generate_dataset_streamed(base_masks, out_dir=None):
    if out_dir is None:
        out_dir = f"dataset_{N}x{N}_final"
    
    os.makedirs(out_dir, exist_ok=True)
    win_lines = expand_lines_via_symmetry(base_masks)
    seen = set()

    print(f"Generating {N}x{N} dataset → {out_dir}")
    print(f"Win lines found: {len(win_lines)}")
    print(f"Layer range: {MIN_LAYER} to {MAX_LAYER}\n")
    total_start = time.time()

    total_layers = MAX_LAYER - MIN_LAYER + 1
    for layer_i, layer in enumerate(range(MIN_LAYER, MAX_LAYER + 1), 1):
        filename = os.path.join(out_dir, f"layer_{layer:02d}.bin")
        init_layer_file(filename)
        count = 0
        start = time.time()

        print(f"[{layer_i}/{total_layers}] Layer {layer:02d} started...")
        inner_total = len(win_lines) * 2
        processed = 0
        last_update = time.time()

        for line in win_lines:
            for winner in (1, 2):
                for board in enumerate_wins_for_line(line, winner, win_lines, layer):
                    cb = canonical_board(board)
                    if cb in seen:
                        continue
                    seen.add(cb)
                    append_record(filename, cb)
                    count += 1

                processed += 1
                # cập nhật tiến trình mỗi 5 giây hoặc khi xong 10%
                if time.time() - last_update > 5 or processed % (inner_total // 10 + 1) == 0:
                    pct = processed / inner_total * 100
                    elapsed = time.time() - start
                    print(f"  progress: {pct:5.1f}% ({processed}/{inner_total}), {elapsed:.1f}s elapsed")
                    last_update = time.time()

        finalize_count(filename)
        elapsed = time.time() - start
        print(f"Layer {layer:02d}: {count} records ({elapsed:.1f}s)\n")

    total_elapsed = time.time() - total_start
    print(f"✓ Done. Total time: {total_elapsed/60:.1f} min ({total_elapsed:.1f} s)")

def enumerate_wins_for_line(line: Tuple[int,...], winner: int, win_lines: List[Tuple[int,...]], ply_min=0, ply_max=999) -> Iterable[List[int]]:
    """
    Sinh tất cả bàn cờ thắng hợp pháp cho 'winner' trên 'line',
    nhưng chỉ với những cấu hình “canonical line” → giảm đối xứng trước khi sinh.
    """
    # **Symmetry-breaking trước**: chỉ sinh nếu line là đại diện canonical trong lớp phép biến đổi
    # Tức: transform tất cả lines qua TRANSFORMS và chỉ chấp nhận `line` nếu nó là min theo thứ tự nào đó.
    line_canon_idx = min(
        tuple(sorted(rc_to_i(*T(*i_to_rc(i))) for i in line))
        for T in TRANSFORMS
    )
    if tuple(sorted(line)) != line_canon_idx:
        return  # chỉ giữ line canonical
    
    if winner == 1:
        ply_values = range(max(MIN_LAYER, ply_min), min(MAX_LAYER + 1, ply_max + 1), 2)
    else:
        ply_values = range(max(MIN_LAYER + 1, ply_min), min(MAX_LAYER + 1, ply_max + 1), 2)

    for T in ply_values:
        if winner == 1:
            x_total = (T + 1) // 2
            o_total = (T - 1) // 2
            extra_w = x_total - N
            extra_other = o_total
            w_val, other_val = 1, 2
        else:
            x_total = T // 2
            o_total = T // 2
            extra_w = o_total - N
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

                # --- Bộ kiểm tra hợp pháp ---
                if winners(b, win_lines) != {winner}:
                    continue
                if not valid_turn_counts(b, win_lines):
                    continue
                if not min_ply_ok(b, win_lines):
                    continue
                if not is_terminal_reachable(b, winner, win_lines):
                    continue

                # --- trước khi yield, có thể canonicalize board và kiểm tra chỉ yield nếu board == canonical_board(board)
                b_tuple = tuple(b)
                cb = canonical_board(b_tuple)
                if b_tuple != cb:
                    continue  # chỉ yield cấu hình canonical (ký hiệu theo Liberti / Katebi)
                yield b_tuple

# ====== Generate Base Masks for NxN ======
def generate_base_masks(n):
    """
    Tạo base masks tối thiểu cho bảng NxN.
    Nhờ D8 symmetry (rotation + reflection), chỉ cần:
    - Nếu n chẵn: n // 2 hàng ngang
    - Nếu n lẻ: n // 2 + 1 hàng ngang (có hàng giữa)
    - 1 đường chéo chính
    
    Ví dụ:
    - 4x4 (chẵn): 4//2 = 2 hàng → 2 hàng + 1 chéo = 3 cases
    - 5x5 (lẻ): 5//2 + 1 = 3 hàng → 3 hàng + 1 chéo = 4 cases
    
    Mỗi case có 2 variants (player 1 và player 2)
    """
    size = n * n
    masks = []
    
    # Số hàng ngang cần
    if n % 2 == 0:  # n chẵn
        num_rows = n // 2
    else:  # n lẻ (có hàng giữa)
        num_rows = n // 2 + 1
    
    # Hàng ngang cho player 1
    for row_idx in range(num_rows):
        row = [1 if row_idx * n <= i < (row_idx + 1) * n else 0 for i in range(size)]
        masks.append(row)
    
    # Đường chéo chính cho player 1
    diag1 = [1 if i % (n + 1) == 0 and i < size else 0 for i in range(size)]
    masks.append(diag1)
    
    # Tương tự cho player 2
    for row_idx in range(num_rows):
        row = [2 if row_idx * n <= i < (row_idx + 1) * n else 0 for i in range(size)]
        masks.append(row)
    
    # Đường chéo chính cho player 2
    diag1_p2 = [2 if i % (n + 1) == 0 and i < size else 0 for i in range(size)]
    masks.append(diag1_p2)
    
    return masks

BASE_MASKS = generate_base_masks(N)

print("Base masks generated:")
for i, mask in enumerate(BASE_MASKS):
    print(f"Mask {i+1}: {mask}")
print()

# ====== RUN ======
if __name__ == "__main__":
    generate_dataset_streamed(BASE_MASKS)