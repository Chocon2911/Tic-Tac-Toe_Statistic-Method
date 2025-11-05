import itertools
import time
from typing import List, Tuple, Iterable, Set

# ===== CONFIG =====
N = 4  # Đổi sang 4 cho 4x4
SIZE = N * N
ALL_CELLS = tuple(range(SIZE))

def rc_to_i(r: int, c: int) -> int: return r * N + c
def i_to_rc(i: int) -> Tuple[int, int]: return divmod(i, N)

# ===== D4 Symmetry =====
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
    for i, v in enumerate(board):
        if v == 0: 
            continue
        r, c = i_to_rc(i)
        r2, c2 = T(r, c)
        out[rc_to_i(r2, c2)] = v
    return tuple(out)

def canonical_board(board: List[int]) -> Tuple[int, ...]:
    return min(transform_board(board, T) for T in TRANSFORMS)

# ===== Winning lines =====
def generate_win_lines() -> List[Tuple[int, ...]]:
    lines = []
    for r in range(N):
        lines.append(tuple(rc_to_i(r, c) for c in range(N)))  # hàng
    for c in range(N):
        lines.append(tuple(rc_to_i(r, c) for r in range(N)))  # cột
    lines.append(tuple(rc_to_i(i, i) for i in range(N)))      # chéo chính
    lines.append(tuple(rc_to_i(i, N-1-i) for i in range(N)))  # chéo phụ
    return lines

WIN_LINES = generate_win_lines()

# ===== Helpers (win / đếm / luật lượt đi) =====
def has_winner(board: List[int], player: int) -> bool:
    return any(all(board[i] == player for i in line) for line in WIN_LINES)

def winners(board: List[int]) -> Set[int]:
    s = set()
    if has_winner(board, 1): s.add(1)
    if has_winner(board, 2): s.add(2)
    return s

def count_pieces(board: List[int]) -> Tuple[int,int]:
    x = sum(1 for v in board if v == 1)
    o = sum(1 for v in board if v == 2)
    return x, o

def valid_turn_counts(board: List[int]) -> bool:
    x, o = count_pieces(board)
    if not (x == o or x == o + 1):
        return False
    if has_winner(board, 2) and x != o:
        return False
    if has_winner(board, 1) and x != o + 1:
        return False
    return True

def min_ply_ok(board: List[int]) -> bool:
    """lọc nhanh theo ngưỡng tối thiểu để có thể thắng: X tại ply = 2N-1, O tại ply = 2N"""
    x, o = count_pieces(board)
    if has_winner(board, 1) and (2*x - 1) < (2*N - 1):
        return False
    if has_winner(board, 2) and (2*o) < (2*N):
        return False
    return True

def is_terminal_reachable(board: List[int], winner: int) -> bool:
    """
    Kiểm tra có thể coi đây là trạng thái kết thúc: chỉ một bên thắng
    và có một "nước cuối" trên một line thắng sao cho bỏ quân đó đi thì:
    - lượt đếm hợp lệ của bàn trước
    - không còn ai thắng ở bàn trước
    """
    if winners(board) != {winner}:
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

    # thử bỏ 1 quân thắng trên một line thắng để về bàn "trước"
    for line in WIN_LINES:
        if all(board[i] == winner for i in line):
            for i in line:
                if board[i] != winner: 
                    continue
                prev = board.copy()
                prev[i] = 0
                if count_pieces(prev) != (prev_x, prev_o):
                    continue
                if winners(prev):  # bàn trước vẫn có người thắng -> không phải nước cuối
                    continue
                if not valid_turn_counts(prev):  # bàn trước phải hợp lệ luật lượt đi
                    continue
                return True
    return False

# ===== Sinh các bàn thắng (đúng luật & terminal) theo ply =====
def enumerate_wins_unique_symmetry() -> Tuple[Set[Tuple[int,...]], Set[Tuple[int,...]]]:
    """
    Trả về (x_wins, o_wins) là các canonical boards unique theo symmetry.
    Ý tưởng: duyệt theo ply (lượt tổng) để ràng buộc số quân X/O,
    sau đó chọn thêm quân ngoài line thắng (cả của người thắng và đối thủ).
    """
    x_wins: Set[Tuple[int,...]] = set()
    o_wins: Set[Tuple[int,...]] = set()

    MIN_X_PLY = 2*N - 1  # X thắng lần đầu có thể ở ply này (nước lẻ)
    MIN_O_PLY = 2*N      # O thắng lần đầu có thể ở ply này (nước chẵn)
    MAX_PLY   = SIZE

    for line in WIN_LINES:
        remaining = tuple(i for i in ALL_CELLS if i not in line)

        # ----- X thắng (ply lẻ) -----
        for T in range(MIN_X_PLY, MAX_PLY + 1, 2):
            x_total = (T + 1) // 2
            o_total = (T - 1) // 2
            extra_x = x_total - N    # ngoài line
            extra_o = o_total
            if extra_x < 0 or extra_o < 0: 
                continue
            if extra_x + extra_o > len(remaining): 
                continue

            # chọn vị trí thêm của X trước, rồi O
            for extra_x_cells in itertools.combinations(remaining, extra_x):
                rem2 = [i for i in remaining if i not in extra_x_cells]
                if extra_o > len(rem2): 
                    continue
                for o_cells in itertools.combinations(rem2, extra_o):
                    b = [0]*SIZE
                    # line thắng của X
                    for i in line: b[i] = 1
                    # thêm X ngoài line
                    for i in extra_x_cells: b[i] = 1
                    # thêm O
                    # (nếu trùng, sẽ bị lọc bởi kiểm tra sau vì không thể đặt 2 quân 1 ô)
                    conflict = any(b[i] != 0 for i in o_cells)
                    if conflict: 
                        continue
                    for i in o_cells: b[i] = 2

                    # Kiểm tra hợp lệ/terminal
                    if winners(b) != {1}: 
                        continue
                    if not valid_turn_counts(b): 
                        continue
                    if not min_ply_ok(b): 
                        continue
                    if not is_terminal_reachable(b, 1): 
                        continue

                    x_wins.add(canonical_board(b))

        # ----- O thắng (ply chẵn) -----
        for T in range(MIN_O_PLY, MAX_PLY + 1, 2):
            x_total = T // 2
            o_total = T // 2
            extra_o = o_total - N
            extra_x = x_total
            if extra_o < 0 or extra_x < 0: 
                continue
            if extra_x + extra_o > len(remaining): 
                continue

            for extra_o_cells in itertools.combinations(remaining, extra_o):
                rem2 = [i for i in remaining if i not in extra_o_cells]
                if extra_x > len(rem2): 
                    continue
                for x_cells in itertools.combinations(rem2, extra_x):
                    b = [0]*SIZE
                    for i in line: b[i] = 2
                    for i in extra_o_cells: b[i] = 2
                    if any(b[i] != 0 for i in x_cells):
                        continue
                    for i in x_cells: b[i] = 1

                    if winners(b) != {2}: 
                        continue
                    if not valid_turn_counts(b): 
                        continue
                    if not min_ply_ok(b): 
                        continue
                    if not is_terminal_reachable(b, 2): 
                        continue

                    o_wins.add(canonical_board(b))

    return x_wins, o_wins

# ===== Sinh hòa (bàn đầy, không ai thắng) =====
def enumerate_draws_unique_symmetry() -> Set[Tuple[int,...]]:
    """
    Hòa chuẩn trong Tic-Tac-Toe: bàn đầy và không có người thắng.
    Duyệt 2^SIZE cấu hình full-board (X/O), kiểm tra luật lượt đi hợp lệ.
    """
    draws: Set[Tuple[int,...]] = set()
    for pattern in itertools.product((1,2), repeat=SIZE):
        if has_winner(list(pattern), 1) or has_winner(list(pattern), 2):
            continue
        x = sum(v == 1 for v in pattern)
        o = SIZE - x
        # Bàn đầy -> lượt đi cuối là X nếu SIZE lẻ, còn O nếu SIZE chẵn.
        # Điều kiện lượt đi hợp lệ chung:
        if x == o or x == o + 1:
            draws.add(canonical_board(list(pattern)))
    return draws

# ===== MAIN =====
if __name__ == "__main__":
    print("="*60)
    print(f" KẾT QUẢ {N}x{N} TIC-TAC-TOE (unique theo symmetry)")
    print("="*60)
    t0 = time.time()
    x_wins, o_wins = enumerate_wins_unique_symmetry()
    draws = enumerate_draws_unique_symmetry()
    dt = time.time() - t0

    total_unique = len(x_wins) + len(o_wins) + len(draws)
    print(f"Tổng số trạng thái kết thúc (unique): {total_unique:,}")
    print(f"  - X thắng: {len(x_wins):,}")
    print(f"  - O thắng: {len(o_wins):,}")
    print(f"  - Hòa:     {len(draws):,}")
    print(f"Thời gian: {dt:.2f}s")
    print("="*60)
