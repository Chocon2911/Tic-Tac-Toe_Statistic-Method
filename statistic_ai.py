import requests
import numpy as np
import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

#=========================================Data Type==========================================
class Position:
    def __init__(self, i: int, j: int, val: str):
        """
        i, j: 1-indexed row, column
        val: 'X', 'O', ... hoặc bất kỳ ký tự nào
        """
        self.i = i
        self.j = j
        self.val = val

class Board:
    def __init__(self, positions: list, size: int = 5, layer: int = 0, win_actor: str = ''):
        """
        positions: list of Position objects
        size: kích thước bàn cờ (default 5x5)
        layer: lớp hiện tại (số nước đã đi)
        win_actor: người thắng ('X', 'O', '' nếu chưa kết thúc hoặc hòa)
        """
        self.size = size
        self.positions = positions
        self.layer = layer
        self.win_actor = win_actor
        self.board = np.full((size, size), '.', dtype=str)  # ô trống là '.'
        for pos in self.positions:
            self.board[pos.i-1, pos.j-1] = pos.val  # chuyển 1-index → 0-index

    def canonical_form(self) -> 'Board':
        """
        Trả về Board ở dạng canonical (xoay/flip tối ưu hóa đối xứng)
        """
        boards = []

        for k in range(4):  # rotate 0, 90, 180, 270
            rot = np.rot90(self.board, k)
            boards.append(rot)
            boards.append(np.fliplr(rot))  # flip ngang
            boards.append(np.flipud(rot))  # flip dọc

        # Chuyển tất cả biến thể thành string row-major
        board_strings = [''.join(b.flatten()) for b in boards]

        # Chọn canonical form: string nhỏ nhất theo lex order
        min_string = min(board_strings)
        min_index = board_strings.index(min_string)
        
        # Lấy numpy array tương ứng
        canonical_array = boards[min_index]
        
        # Tạo danh sách Position từ canonical array
        canonical_positions = []
        for i in range(self.size):
            for j in range(self.size):
                if canonical_array[i, j] != '.':
                    canonical_positions.append(
                        Position(i+1, j+1, canonical_array[i, j])
                    )
        
        # Trả về Board mới với canonical positions
        return Board(canonical_positions, self.size, self.layer, self.win_actor)
    
    def add_pos(self, pos: Position):
        """
        Thêm một Position vào board
        """
        self.positions.append(pos)
        self.board[pos.i-1, pos.j-1] = pos.val
        self.layer += 1

    def check_win(self) -> str:
        """
        Kiểm tra ai thắng trên board hiện tại
        Trả về 'X', 'O', hoặc '' (chưa có người thắng)
        """
        # Kiểm tra hàng ngang
        for i in range(self.size):
            for j in range(self.size - 4):
                if self.board[i, j] != '.' and \
                   all(self.board[i, j+k] == self.board[i, j] for k in range(5)):
                    return self.board[i, j]
        
        # Kiểm tra hàng dọc
        for i in range(self.size - 4):
            for j in range(self.size):
                if self.board[i, j] != '.' and \
                   all(self.board[i+k, j] == self.board[i, j] for k in range(5)):
                    return self.board[i, j]
        
        # Kiểm tra đường chéo chính (\)
        for i in range(self.size - 4):
            for j in range(self.size - 4):
                if self.board[i, j] != '.' and \
                   all(self.board[i+k, j+k] == self.board[i, j] for k in range(5)):
                    return self.board[i, j]
        
        # Kiểm tra đường chéo phụ (/)
        for i in range(4, self.size):
            for j in range(self.size - 4):
                if self.board[i, j] != '.' and \
                   all(self.board[i-k, j+k] == self.board[i, j] for k in range(5)):
                    return self.board[i, j]
        
        return ''

    def is_full(self) -> bool:
        """Kiểm tra bàn cờ đã đầy chưa"""
        return not np.any(self.board == '.')

    def __str__(self):
        board_str = "  " + " ".join(str(i+1) for i in range(self.size)) + "\n"
        for i in range(self.size):
            board_str += str(i+1) + " " + " ".join(self.board[i]) + "\n"
        if self.win_actor:
            board_str += f"Winner: {self.win_actor}\n"
        return board_str

#==========================================Database==========================================
CLICKHOUSE_HTTP = "http://localhost:8123"
CLICKHOUSE_USER = "default"
CLICKHOUSE_PASS = "admin"
DATABASE = "tictactoe"

# ✅ Connection pooling
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(
    pool_connections=100,
    pool_maxsize=100,
    max_retries=3
)
session.mount('http://', adapter)

# ✅ Query cache
QUERY_CACHE = {}
MAX_CACHE_SIZE = 10000

def query_by_positions_optimized(table: str, positions: list):
    """
    Truy xuất từ ClickHouse với tối ưu hóa:
    - Chỉ SELECT 2 columns cần thiết
    - Sử dụng connection pooling
    """
    if not positions:
        raise ValueError("positions không được rỗng")

    # Cache key
    cache_key = (table, tuple(sorted([(p['i'] if isinstance(p, dict) else p[0], 
                                       p['j'] if isinstance(p, dict) else p[1]) 
                                      for p in positions])))
    
    if cache_key in QUERY_CACHE:
        return QUERY_CACHE[cache_key]

    # Chuyển danh sách thành điều kiện WHERE
    conditions = []
    for pos in positions:
        if isinstance(pos, dict):
            i, j = pos['i'], pos['j']
        else:  # tuple/list
            i, j = pos
        col_name = f"i{i}{j}"
        conditions.append(f"{col_name} != ''")
    where_clause = " AND ".join(conditions)

    # ✅ Chỉ SELECT 2 columns cần thiết
    sql = f"SELECT canonical_form, win_actor FROM {DATABASE}.{table} WHERE {where_clause}"

    response = session.post(  # ✅ Dùng session pool
        CLICKHOUSE_HTTP,
        params={
            "user": CLICKHOUSE_USER,
            "password": CLICKHOUSE_PASS,
        },
        data=sql,
        timeout=5
    )

    if response.status_code != 200:
        return []

    if not response.text.strip():
        return []

    # Parse TSV
    rows = response.text.strip().split("\n")
    data = [row.split("\t") for row in rows]
    
    # Cache result
    if len(QUERY_CACHE) < MAX_CACHE_SIZE:
        QUERY_CACHE[cache_key] = data
    
    return data


def batch_query_parallel(table: str, conditions_list: list, max_workers: int = 30):
    """
    Query song song nhiều conditions
    
    Args:
        table: Tên bảng
        conditions_list: List các conditions [[{i,j},...], [{i,j},...]]
        max_workers: Số thread song song
    
    Returns:
        List kết quả tương ứng với từng condition
    """
    results = [None] * len(conditions_list)
    
    def query_one(index, conditions):
        try:
            return index, query_by_positions_optimized(table, conditions)
        except Exception as e:
            return index, []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(query_one, i, cond): i 
            for i, cond in enumerate(conditions_list)
        }
        
        for future in as_completed(futures):
            index, result = future.result()
            results[index] = result
    
    return results

#=======================================Transformation=======================================
def get_transform_mapping(original_board: Board, canonical_board: Board) -> dict:
    """
    Tìm transformation từ original board sang canonical board.
    Trả về dict chứa thông tin transformation.
    """
    size = original_board.size
    
    # Thử tất cả các transformation
    transformations = []
    for k in range(4):  # rotate 0, 90, 180, 270
        rot = np.rot90(original_board.board, k)
        transformations.append(('rot', k, rot))
        transformations.append(('rot_fliplr', k, np.fliplr(rot)))
        transformations.append(('rot_flipud', k, np.flipud(rot)))
    
    # Tìm transformation khớp với canonical
    for trans_type, rotation, transformed in transformations:
        if np.array_equal(transformed, canonical_board.board):
            return {
                'type': trans_type,
                'rotation': rotation,
                'size': size
            }
    
    # Nếu không tìm thấy, trả về identity
    return {'type': 'identity', 'rotation': 0, 'size': size}


def reverse_transform(canonical_pos: tuple, transform_map: dict, size: int) -> tuple:
    """
    Map position từ canonical board về original board.
    
    Args:
        canonical_pos: (i, j) trên canonical board (1-indexed)
        transform_map: Dict chứa thông tin transformation
        size: Kích thước board
    
    Returns:
        (i, j) trên original board (1-indexed)
    """
    if transform_map['type'] == 'identity':
        return canonical_pos
    
    i, j = canonical_pos[0] - 1, canonical_pos[1] - 1  # Chuyển về 0-indexed
    
    # Tạo ma trận test
    test_board = np.full((size, size), '.', dtype=str)
    test_board[i, j] = 'T'  # Đánh dấu vị trí
    
    trans_type = transform_map['type']
    rotation = transform_map['rotation']
    
    # Apply transformation ngược
    if trans_type == 'rot_flipud':
        test_board = np.flipud(test_board)
    elif trans_type == 'rot_fliplr':
        test_board = np.fliplr(test_board)
    
    # Rotate ngược (4-k rotations)
    test_board = np.rot90(test_board, 4 - rotation)
    
    # Tìm vị trí 'T' trong board gốc
    pos = np.where(test_board == 'T')
    if len(pos[0]) > 0:
        return (pos[0][0] + 1, pos[1][0] + 1)  # Chuyển về 1-indexed
    
    return canonical_pos  # Fallback

#========================================Heuristic=========================================
def get_board_score(board: Board, player: str) -> float:
    """Heuristic scoring cho board"""
    score = 0.0
    opponent = 'O' if player == 'X' else 'X'
    
    # Đếm sequences
    for length in [4, 3, 2]:
        player_seq = count_sequences(board, player, length)
        opponent_seq = count_sequences(board, opponent, length)
        
        score += player_seq * (length ** 3)
        score -= opponent_seq * (length ** 2)
    
    return score


def count_sequences(board: Board, player: str, length: int) -> int:
    """Đếm số sequences có độ dài length"""
    count = 0
    size = board.size
    
    # Ngang
    for i in range(size):
        for j in range(size - length + 1):
            seq = [board.board[i, j+k] for k in range(length)]
            if seq.count(player) == length and '.' not in seq:
                count += 1
    
    # Dọc
    for i in range(size - length + 1):
        for j in range(size):
            seq = [board.board[i+k, j] for k in range(length)]
            if seq.count(player) == length and '.' not in seq:
                count += 1
    
    return count

#==========================================AI Logic==========================================
def next_best_move(board: Board, player: str) -> tuple:
    """
    Tính nước đi tốt nhất tiếp theo cho player ('X' hoặc 'O') trên board hiện tại.
    Sử dụng BFS có điều kiện với parallel queries.
    Trả về (i, j) 1-indexed của nước đi tốt nhất.
    """
    start_time = time.time()
    total_queries = 0
    total_bytes = 0
    
    # Clear cache nếu quá lớn
    global QUERY_CACHE
    if len(QUERY_CACHE) > MAX_CACHE_SIZE:
        QUERY_CACHE.clear()
    
    # === Canonical hóa board trước khi xử lý ===
    canonical_board = board.canonical_form()
    transform_map = get_transform_mapping(board, canonical_board)
    
    boards_by_layer = {}
    opponent = 'O' if player == 'X' else 'X'
    
    # === Layer 0: Nước đi đầu tiên ===
    curr_layer = canonical_board.layer + 1
    curr_player = 'X' if (curr_layer % 2) == 1 else 'O'
    
    if curr_player != player:
        print(f"Cảnh báo: Không phải lượt của {player}")
        return None
    
    unique_moves = get_unique_moves(canonical_board, curr_player)
    
    if not unique_moves:
        return None
    
    boards_by_layer[0] = []
    
    if canonical_board.layer < 9:  # Chưa cần query DB
        for move in unique_moves:
            new_board = Board(copy.deepcopy(canonical_board.positions), canonical_board.size, curr_layer)
            new_board.add_pos(Position(move[0], move[1], curr_player))
            
            winner = new_board.check_win()
            if winner == player:
                original_move = reverse_transform(move, transform_map, board.size)
                return original_move
            
            boards_by_layer[0].append((new_board, move))
    else:
        # ✅ PARALLEL QUERIES cho layer 0
        base_condition = [{'i': pos.i, 'j': pos.j} for pos in canonical_board.positions]
        
        conditions_list = []
        for move in unique_moves:
            condition = base_condition.copy()
            condition.append({'i': move[0], 'j': move[1]})
            conditions_list.append(condition)
        
        # Query tất cả song song
        all_results = batch_query_parallel(
            f"ttt_{canonical_board.size}_l{curr_layer}",
            conditions_list,
            max_workers=30
        )
        
        total_queries += len(conditions_list)
        
        # Process results
        for move, data in zip(unique_moves, all_results):
            if not data:
                continue
            
            total_bytes += len(str(data))
            
            for row in data:
                result_board = data_to_board_light(row, canonical_board.size, curr_layer)
                
                if result_board.win_actor == player:
                    original_move = reverse_transform(move, transform_map, board.size)
                    elapsed = time.time() - start_time
                    speed_mb_s = (total_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0
                    print(f"⚡ Stats: {total_queries} queries, {elapsed:.2f}s, {speed_mb_s:.1f} MB/s")
                    return original_move
                
                if result_board.win_actor != opponent:
                    boards_by_layer[0].append((result_board, move))
    
    # Adaptive depth
    if board.layer < 5:
        max_depth = 8
    elif board.layer < 15:
        max_depth = 10
    else:
        max_depth = 16
    
    # === BFS với parallel queries và pruning ===
    for layer_offset in range(1, min(max_depth, 26 - canonical_board.layer)):
        curr_layer = canonical_board.layer + layer_offset + 1
        curr_player = 'X' if (curr_layer % 2) == 1 else 'O'
        is_player_turn = (curr_player == player)
        
        boards_by_layer[layer_offset] = []
        prev_layer_boards = boards_by_layer.get(layer_offset - 1, [])
        
        if not prev_layer_boards:
            break
        
        # ✅ Pruning: chỉ giữ top boards
        if len(prev_layer_boards) > 50:
            prev_layer_boards = sorted(
                prev_layer_boards,
                key=lambda x: get_board_score(x[0], player),
                reverse=True
            )[:50]
        
        # Collect all queries
        all_conditions = []
        query_metadata = []
        
        for prev_board, first_move_tuple in prev_layer_boards:
            if prev_board.layer < 9:
                unique_moves = get_unique_moves(prev_board, curr_player)
                for move in unique_moves:
                    new_board = Board(copy.deepcopy(prev_board.positions), canonical_board.size, curr_layer)
                    new_board.add_pos(Position(move[0], move[1], curr_player))
                    
                    winner = new_board.check_win()
                    
                    if is_player_turn and winner == player:
                        original_move = reverse_transform(first_move_tuple, transform_map, board.size)
                        elapsed = time.time() - start_time
                        speed_mb_s = (total_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0
                        print(f"⚡ Stats: {total_queries} queries, {elapsed:.2f}s, {speed_mb_s:.1f} MB/s")
                        return original_move
                    
                    if winner != opponent:
                        boards_by_layer[layer_offset].append((new_board, first_move_tuple))
            else:
                base_condition = [{'i': pos.i, 'j': pos.j} for pos in prev_board.positions]
                unique_moves = get_unique_moves(prev_board, curr_player)
                
                for move in unique_moves:
                    condition = base_condition.copy()
                    condition.append({'i': move[0], 'j': move[1]})
                    all_conditions.append(condition)
                    query_metadata.append((prev_board, first_move_tuple, move))
        
        # ✅ Query tất cả song song
        if all_conditions:
            all_results = batch_query_parallel(
                f"ttt_{canonical_board.size}_l{curr_layer}",
                all_conditions,
                max_workers=30
            )
            
            total_queries += len(all_conditions)
            
            # Process results
            for (prev_board, first_move_tuple, move), data in zip(query_metadata, all_results):
                if not data:
                    continue
                
                total_bytes += len(str(data))
                
                for row in data:
                    result_board = data_to_board_light(row, canonical_board.size, curr_layer)
                    
                    if is_player_turn:
                        if result_board.win_actor == player:
                            original_move = reverse_transform(first_move_tuple, transform_map, board.size)
                            elapsed = time.time() - start_time
                            speed_mb_s = (total_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0
                            print(f"⚡ Stats: {total_queries} queries, {elapsed:.2f}s, {speed_mb_s:.1f} MB/s")
                            return original_move
                        
                        if result_board.win_actor != opponent:
                            boards_by_layer[layer_offset].append((result_board, first_move_tuple))
                    else:
                        if result_board.win_actor != opponent:
                            boards_by_layer[layer_offset].append((result_board, first_move_tuple))
    
    # Fallback
    elapsed = time.time() - start_time
    speed_mb_s = (total_bytes / 1024 / 1024) / elapsed if elapsed > 0 else 0
    print(f"⚡ Stats: {total_queries} queries, {elapsed:.2f}s, {speed_mb_s:.1f} MB/s")
    
    if boards_by_layer.get(0):
        canonical_move = boards_by_layer[0][0][1]
        original_move = reverse_transform(canonical_move, transform_map, board.size)
        return original_move
    
    if unique_moves:
        canonical_move = unique_moves[0]
        original_move = reverse_transform(canonical_move, transform_map, board.size)
        return original_move
    
    return None

#=======================================data to board========================================
def data_to_board_light(row: list, size: int = 5, layer: int = 0) -> Board:
    """
    Parse board nhẹ - CHỈ lấy canonical_form và win_actor
    Không parse 25 cells vì không cần thiết
    """
    if not row or len(row) < 2:
        raise ValueError("Row data không hợp lệ")
    
    canonical_form = row[0].strip()
    win_actor = row[1].strip()
    
    # Trả về board minimal
    board = Board([], size, layer, win_actor)
    board._canonical_form = canonical_form
    
    return board


def data_to_board(row: list, size: int = 5, layer: int = 0) -> Board:
    """
    Chuyển đổi một row từ ClickHouse thành Board object đầy đủ.
    Dùng khi cần positions.
    """
    if not row or len(row) < 2:
        raise ValueError("Row data không hợp lệ")
    
    # Index 0: canonical_form
    # Index 1: win_actor
    win_actor = row[1].strip() if len(row) > 1 else ''
    
    # Parse positions từ i11, i12, ..., i55
    positions = []
    cell_index = 2
    
    for i in range(1, size + 1):
        for j in range(1, size + 1):
            if cell_index < len(row):
                val = row[cell_index].strip()
                if val and val != '' and val != '.':
                    positions.append(Position(i, j, val))
            cell_index += 1
    
    if layer == 0:
        layer = len(positions)
    
    return Board(positions, size, layer, win_actor)

#========================================unique move=========================================
def get_unique_moves(board: Board, player: str) -> list:
    """
    Trả về danh sách các nước đi duy nhất (i, j) 1-indexed cho player trên board hiện tại,
    áp dụng tối ưu hóa đối xứng.
    """
    size = board.size
    unique_moves = {}

    for i in range(size):
        for j in range(size):
            if board.board[i, j] == '.':
                # Thử đặt player tại ô (i,j)
                board.board[i, j] = player
                # Tính canonical form sau nước đi
                canon_board = board.canonical_form()
                # Chuyển board thành string để làm key
                canon_string = ''.join(canon_board.board.flatten())
                # Lưu vào dict: chỉ giữ một move cho mỗi canonical form
                if canon_string not in unique_moves:
                    unique_moves[canon_string] = (i+1, j+1)  # 1-indexed
                # Reset ô
                board.board[i, j] = '.'

    return list(unique_moves.values())

#==========================================Game Play=========================================
def play_game():
    """
    Chơi game Tic-Tac-Toe 5x5
    """
    print("=== TIC-TAC-TOE 5x5 ===")
    print("Chọn chế độ:")
    print("1. Người vs Người")
    print("2. Người vs AI")
    print("3. AI vs AI")
    
    mode = input("Nhập lựa chọn (1/2/3): ").strip()
    
    board = Board([], size=5, layer=0)
    current_player = 'X'
    
    while True:
        print("\n" + "="*30)
        print(board)
        print(f"Lượt: {current_player}")
        
        # Kiểm tra thắng
        winner = board.check_win()
        if winner:
            print(f"\n🎉 {winner} THẮNG! 🎉")
            break
        
        # Kiểm tra hòa
        if board.is_full():
            print("\n🤝 HÒA! 🤝")
            break
        
        # Lấy nước đi
        if mode == '1':  # Người vs Người
            move = get_human_move(board)
        elif mode == '2':  # Người vs AI
            if current_player == 'X':
                move = get_human_move(board)
            else:
                print("AI đang suy nghĩ...")
                move = next_best_move(board, current_player)
                if move:
                    print(f"AI chọn: ({move[0]}, {move[1]})")
        else:  # AI vs AI
            print(f"AI {current_player} đang suy nghĩ...")
            move = next_best_move(board, current_player)
            if move:
                print(f"AI {current_player} chọn: ({move[0]}, {move[1]})")
            input("Nhấn Enter để tiếp tục...")
        
        if not move:
            print("Không có nước đi hợp lệ!")
            break
        
        # Thực hiện nước đi
        board.add_pos(Position(move[0], move[1], current_player))
        
        # Đổi lượt
        current_player = 'O' if current_player == 'X' else 'X'
    
    print("\n" + "="*30)
    print("Game Over!")

def get_human_move(board: Board) -> tuple:
    """Lấy nước đi từ người chơi"""
    while True:
        try:
            move_input = input("Nhập nước đi (i j): ").strip()
            i, j = map(int, move_input.split())
            
            if 1 <= i <= board.size and 1 <= j <= board.size:
                if board.board[i-1, j-1] == '.':
                    return (i, j)
                else:
                    print("Ô này đã có quân cờ!")
            else:
                print(f"Vui lòng nhập trong khoảng 1-{board.size}!")
        except:
            print("Định dạng không hợp lệ! Nhập: i j (VD: 3 3)")

#============================================Main============================================
if __name__ == "__main__":
    # Test canonical transformation
    print("=== Test Canonical Transformation ===")
    positions = [Position(1, 1, 'X'), Position(5, 5, 'O')]
    board = Board(positions, size=5, layer=2)
    
    print("Original board:")
    print(board)
    
    canonical = board.canonical_form()
    print("Canonical board:")
    print(canonical)
    
    transform_map = get_transform_mapping(board, canonical)
    print(f"Transform map: {transform_map}")
    
    # Test reverse transform
    test_pos = (3, 3)
    original_pos = reverse_transform(test_pos, transform_map, 5)
    print(f"Canonical pos {test_pos} -> Original pos {original_pos}")
    
    print("\n" + "="*50 + "\n")
    
    # Chơi game
    play_game()