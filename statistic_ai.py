import requests
import numpy as np
import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

#==========================================Database Configuration==========================================
CLICKHOUSE_HTTP = "http://localhost:8123"
CLICKHOUSE_USER = "default"
CLICKHOUSE_PASS = "admin"
DATABASE = "tictactoe"

# Connection pooling
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=50,
    pool_maxsize=50,
    max_retries=2
)
session.mount('http://', adapter)


def execute_query(sql: str) -> int:
    """
    Thực thi SQL query và trả về COUNT
    
    Args:
        sql: SQL query string
        
    Returns:
        Số lượng rows (int)
    """
    try:
        response = session.post(
            CLICKHOUSE_HTTP,
            params={
                "user": CLICKHOUSE_USER,
                "password": CLICKHOUSE_PASS,
                "database": DATABASE
            },
            data=sql,
            timeout=10
        )
        
        if response.status_code != 200:
            print(f"❌ Query error {response.status_code}: {response.text}")
            return 0
        
        result = response.text.strip()
        if not result:
            return 0
        
        return int(result)
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return 0


def get_odd_table_names(move_count: int) -> str:
    """
    Lấy danh sách tên bảng odd (lượt lẻ)
    
    Args:
        move_count: Số nước đã đi (không dùng, chỉ để giữ signature)
        
    Returns:
        String format SQL: ttt_5_l9, ttt_5_l11, ..., ttt_5_l25
    """
    tables = []
    
    # Lấy tất cả các level lẻ từ 9 đến 25
    for level in range(9, 26, 2):  # 9, 11, 13, ..., 25
        tables.append(f"ttt_5_l{level}")
    
    return ", ".join(tables)


def get_even_table_names(move_count: int) -> str:
    """
    Lấy danh sách tên bảng even (lượt chẵn)
    
    Args:
        move_count: Số nước đã đi (không dùng, chỉ để giữ signature)
        
    Returns:
        String format SQL: ttt_5_l10, ttt_5_l12, ..., ttt_5_l24
    """
    tables = []
    
    # Lấy tất cả các level chẵn từ 10 đến 24
    for level in range(10, 25, 2):  # 10, 12, 14, ..., 24
        tables.append(f"ttt_5_l{level}")
    
    return ", ".join(tables)


def build_where_clause(board: list) -> str:
    """
    Xây dựng WHERE clause từ board
    
    Args:
        board: Board hiện tại
        
    Returns:
        WHERE clause string
    """
    n = 5
    conditions = []
    
    for idx, cell in enumerate(board):
        if cell != 0:
            row = (idx // n) + 1  # +1 vì index bắt đầu từ 1
            col = (idx % n) + 1
            col_name = f"i{row}{col}"
            player_mark = 'X' if cell == 1 else 'O'
            conditions.append(f"{col_name} = '{player_mark}'")
    
    return " AND ".join(conditions) if conditions else "1=1"


def query_odd_table(board: list) -> int:
    """
    Query bảng odd (lượt lẻ) - đếm số trận X thắng
    
    Args:
        board: Bảng hiện tại (list of int, size 25)
        
    Returns:
        Số lượng rows có win_actor = 'X'
    """
    move_count = sum(1 for cell in board if cell != 0)
    
    if move_count == 0:
        return 0
    
    where_clause = build_where_clause(board)
    
    # Tạo UNION ALL cho tất cả bảng lẻ
    union_queries = []
    for level in range(9, 26, 2):  # 9, 11, 13, ..., 25
        if level < move_count:
            continue
        table_name = f"ttt_5_l{level}"
        union_queries.append(f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause} AND win_actor = 'X'")

    # Kết hợp và tính tổng
    sql = f"SELECT sum(c) FROM ({' UNION ALL '.join(union_queries)}) AS counts(c)"
    
    return execute_query(sql)


def query_even_table(board: list) -> int:
    """
    Query bảng even (lượt chẵn) - đếm số trận O thắng
    
    Args:
        board: Bảng hiện tại (list of int, size 25)
        
    Returns:
        Số lượng rows có win_actor = 'O'
    """
    move_count = sum(1 for cell in board if cell != 0)
    
    if move_count == 0:
        return 0
    
    where_clause = build_where_clause(board)
    
    # Tạo UNION ALL cho tất cả bảng chẵn
    union_queries = []
    for level in range(10, 25, 2):  # 10, 12, 14, ..., 24
        if level < move_count:
            continue
        table_name = f"ttt_5_l{level}"
        union_queries.append(f"SELECT COUNT(*) FROM {table_name} WHERE {where_clause} AND win_actor = 'O'")
    
    # Kết hợp và tính tổng
    sql = f"SELECT sum(c) FROM ({' UNION ALL '.join(union_queries)}) AS counts(c)"
    
    return execute_query(sql)

#=========================================Symmetric==========================================
N = 5  # Board size constant

# Transformation functions
def t_identity(r, c):
    return (r, c)

def t_rot90(r, c):
    return (c, N-1-r)

def t_rot180(r, c):
    return (N-1-r, N-1-c)

def t_rot270(r, c):
    return (N-1-c, r)

def t_reflect_h(r, c):
    return (N-1-r, c)

def t_reflect_v(r, c):
    return (r, N-1-c)

def t_reflect_main(r, c):
    return (c, r)

def t_reflect_anti(r, c):
    return (N-1-c, N-1-r)


def apply_transformation(board: list, transform_func) -> list:
    """
    Áp dụng transformation function lên board
    
    Args:
        board: Board 1D (25 elements)
        transform_func: Hàm transformation (r,c) -> (r',c')
        
    Returns:
        Board mới sau khi transform
    """
    n = N
    new_board = [0] * (n * n)
    
    for idx in range(n * n):
        r = idx // n
        c = idx % n
        
        # Apply transformation
        new_r, new_c = transform_func(r, c)
        new_idx = new_r * n + new_c
        
        new_board[new_idx] = board[idx]
    
    return new_board


def get_symmetries(board: list) -> list:
    """
    Tạo tất cả các phép biến đổi đối xứng của board 5x5
    Dùng cùng transformations như lúc gen data
    
    Args:
        board: Board hiện tại (list 25 elements)
        
    Returns:
        List các board đối xứng (8 biến đổi)
    """
    transformations = [
        t_identity,
        t_rot90,
        t_rot180,
        t_rot270,
        t_reflect_h,
        t_reflect_v,
        t_reflect_main,
        t_reflect_anti
    ]
    
    symmetries = []
    for transform in transformations:
        sym_board = apply_transformation(board, transform)
        symmetries.append(sym_board)
    
    return symmetries


def canonical_board(board: list) -> list:
    """
    Tìm canonical form của board (form nhỏ nhất theo lexicographic order)
    Giống như lúc gen data
    
    Args:
        board: Board hiện tại
        
    Returns:
        Canonical board
    """
    symmetries = get_symmetries(board)
    
    # Convert to tuples for comparison
    sym_tuples = [tuple(sym) for sym in symmetries]
    
    # Return the lexicographically smallest
    return list(min(sym_tuples))

#==========================================AI Logic==========================================
def best_step(currBoard: list, player: int):
    """
    Tìm nước đi tốt nhất cho AI dựa trên database
    
    Args:
        currBoard: Board hiện tại
        player: Player hiện tại (1 hoặc 2)
        
    Returns:
        Index của nước đi tốt nhất, hoặc -1 nếu không tìm thấy
    """
    start_time = time.time()

    best_move = -1
    win_rate = 0
    
    # Log số ô trống
    empty_cells = sum(1 for cell in currBoard if cell == 0)
    print(f"\n🤔 AI đang suy nghĩ... (Còn {empty_cells} ô trống)")
    
    moves_checked = 0
    moves_with_data = 0

    for i in range(len(currBoard)):
        if currBoard[i] != 0:
            continue

        moves_checked += 1
        newBoard = copy.deepcopy(currBoard)
        newBoard[i] = player

        # Convert to canonical form trước khi query
        canonical = canonical_board(newBoard)
        
        # Query với canonical form
        x_win_count = query_odd_table(canonical)
        o_win_count = query_even_table(canonical)
        total_count = x_win_count + o_win_count
        
        if total_count <= 0:
            continue
        
        moves_with_data += 1
        
        # Tính win rate cho player hiện tại
        win_count = x_win_count if player == 1 else o_win_count
        current_win_rate = win_count / total_count
        
        # Log chi tiết
        row = i // 5
        col = i % 5
        print(f"  Ô [{row},{col}] (idx={i}): win_rate={current_win_rate:.2%} "
              f"(X:{x_win_count}, O:{o_win_count}, total:{total_count})")
        
        if current_win_rate > win_rate:
            win_rate = current_win_rate
            best_move = i

    elapsed_time = time.time() - start_time
    
    if best_move != -1:
        print(f"\n✅ AI chọn ô {best_move} (row={best_move//5}, col={best_move%5})")
        print(f"   Win rate: {win_rate:.2%}")
    else:
        print(f"\n⚠️  Không tìm thấy nước đi tốt trong database")
        # Fallback: chọn ô trống đầu tiên
        for i in range(len(currBoard)):
            if currBoard[i] == 0:
                best_move = i
                break
        if best_move != -1:
            print(f"   Chọn random: ô {best_move} (row={best_move//5}, col={best_move%5})")
    
    print(f"⏱️  Thời gian suy nghĩ: {elapsed_time:.3f}s")
    
    return best_move

#==========================================Game Logic==========================================
def print_board(board: list):
    """In bảng game ra console"""
    n = int(len(board) ** 0.5)
    print("\n  " + "   ".join([str(i) for i in range(n)]))
    print("  " + "----" * n)
    
    for i in range(n):
        row = []
        for j in range(n):
            idx = i * n + j
            cell = board[idx]
            if cell == 0:
                row.append(" ")
            elif cell == 1:
                row.append("X")
            else:
                row.append("O")
        print(f"{i}| {' | '.join(row)} |")
        if i < n - 1:
            print("  " + "----" * n)
    print()


def check_winner(board: list) -> int:
    """
    Kiểm tra người thắng
    
    Returns:
        0: chưa có người thắng
        1: player 1 (X) thắng
        2: player 2 (O) thắng
        -1: hòa
    """
    n = int(len(board) ** 0.5)
    
    # Kiểm tra hàng ngang
    for i in range(n):
        for j in range(n - 4):
            if board[i*n + j] != 0:
                if all(board[i*n + j + k] == board[i*n + j] for k in range(5)):
                    return board[i*n + j]
    
    # Kiểm tra hàng dọc
    for i in range(n - 4):
        for j in range(n):
            if board[i*n + j] != 0:
                if all(board[(i+k)*n + j] == board[i*n + j] for k in range(5)):
                    return board[i*n + j]
    
    # Kiểm tra đường chéo chính
    for i in range(n - 4):
        for j in range(n - 4):
            if board[i*n + j] != 0:
                if all(board[(i+k)*n + (j+k)] == board[i*n + j] for k in range(5)):
                    return board[i*n + j]
    
    # Kiểm tra đường chéo phụ
    for i in range(n - 4):
        for j in range(4, n):
            if board[i*n + j] != 0:
                if all(board[(i+k)*n + (j-k)] == board[i*n + j] for k in range(5)):
                    return board[i*n + j]
    
    # Kiểm tra hòa
    if all(cell != 0 for cell in board):
        return -1
    
    return 0


def play_game():
    """Main game loop"""
    board = [0] * 25  # 5x5 board
    current_player = 1  # 1 = X (Human), 2 = O (AI)
    
    print("=" * 50)
    print("🎮 TIC-TAC-TOE 5x5 - AI vs HUMAN 🎮")
    print("=" * 50)
    print("Bạn là X, AI là O")
    print("Nhiệm vụ: Tạo 5 dấu liên tiếp (ngang/dọc/chéo)")
    print("=" * 50)
    
    move_count = 0
    
    while True:
        print_board(board)
        
        winner = check_winner(board)
        if winner != 0:
            if winner == 1:
                print("🎉 Bạn thắng! Chúc mừng!")
            elif winner == 2:
                print("🤖 AI thắng! Hãy thử lại!")
            else:
                print("🤝 Hòa!")
            break
        
        if current_player == 1:
            # Human turn
            print(f"\n--- Lượt của bạn (X) - Nước đi #{move_count + 1} ---")
            while True:
                try:
                    row = int(input("Nhập hàng (0-4): "))
                    col = int(input("Nhập cột (0-4): "))
                    idx = row * 5 + col
                    
                    if row < 0 or row > 4 or col < 0 or col > 4:
                        print("❌ Vị trí không hợp lệ! Hãy nhập 0-4")
                        continue
                    
                    if board[idx] != 0:
                        print("❌ Ô này đã được đánh! Chọn ô khác")
                        continue
                    
                    board[idx] = 1
                    break
                except ValueError:
                    print("❌ Vui lòng nhập số!")
                except KeyboardInterrupt:
                    print("\n👋 Tạm biệt!")
                    return
        else:
            # AI turn
            print(f"\n--- Lượt của AI (O) - Nước đi #{move_count + 1} ---")
            move = best_step(board, 2)
            
            if move == -1:
                print("❌ AI không thể di chuyển!")
                break
            
            board[move] = 2
        
        current_player = 3 - current_player  # Switch: 1 <-> 2
        move_count += 1
        
        # Pause để dễ theo dõi
        if current_player == 1:
            input("\nNhấn Enter để tiếp tục...")


def play_ai_vs_ai():
    """AI vs AI mode để test"""
    board = [0] * 25
    current_player = 1
    
    print("=" * 50)
    print("🤖 TIC-TAC-TOE 5x5 - AI vs AI 🤖")
    print("=" * 50)
    
    move_count = 0
    
    while True:
        print_board(board)
        
        winner = check_winner(board)
        if winner != 0:
            if winner == 1:
                print("🎉 AI X thắng!")
            elif winner == 2:
                print("🤖 AI O thắng!")
            else:
                print("🤝 Hòa!")
            break
        
        print(f"\n--- Lượt của AI {'X' if current_player == 1 else 'O'} - Nước đi #{move_count + 1} ---")
        move = best_step(board, current_player)
        
        if move == -1:
            print("❌ AI không thể di chuyển!")
            break
        
        board[move] = current_player
        current_player = 3 - current_player
        move_count += 1
        
        time.sleep(1)  # Pause để xem


if __name__ == "__main__":
    print("\n🎮 Chọn chế độ chơi:")
    print("1. Human vs AI")
    print("2. AI vs AI (test mode)")
    
    try:
        choice = input("\nNhập lựa chọn (1 hoặc 2): ").strip()
        
        if choice == "1":
            play_game()
        elif choice == "2":
            play_ai_vs_ai()
        else:
            print("❌ Lựa chọn không hợp lệ!")
    except KeyboardInterrupt:
        print("\n👋 Tạm biệt!")