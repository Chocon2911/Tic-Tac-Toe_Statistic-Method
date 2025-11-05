import struct
import csv

# ====== Config ======
N = 4
SIZE = N * N
RECORD_SIZE = 5
MAGIC = 0x54545434  # "TTT4"

def bits_to_board(bits: int) -> list[int]:
    """Decode 32-bit integer -> list of 16 cells (2 bits per cell)."""
    board = []
    for i in range(SIZE):
        board.append((bits >> (i * 2)) & 3)
    return board

def decode_record(data: bytes):
    """Giải mã 5-byte record -> (board, winner, ply)."""
    board_bits, metadata = struct.unpack('<IB', data)
    board = bits_to_board(board_bits)
    winner = (metadata >> 4) & 0xF
    ply_count = metadata & 0xF
    return board, winner, ply_count

def read_binary_dataset(path: str):
    """Đọc toàn bộ dataset .bin"""
    with open(path, 'rb') as f:
        magic, count, record_size = struct.unpack('<III', f.read(12))
        if magic != MAGIC:
            raise ValueError(f"Magic number không hợp lệ: {hex(magic)}")
        if record_size != RECORD_SIZE:
            raise ValueError(f"Kích thước bản ghi không khớp: {record_size}")

        records = []
        for _ in range(count):
            data = f.read(RECORD_SIZE)
            if len(data) < RECORD_SIZE:
                break
            records.append(decode_record(data))
        return records

def convert_bin_to_csv(input_path: str, output_path: str):
    """Convert .bin file to .csv file"""
    records = read_binary_dataset(input_path)
    print(f"Read {len(records)} records from {input_path}")

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"cell_{i:02d}" for i in range(SIZE)] + ["winner", "ply"]
        writer.writerow(header)
        for board, winner, ply in records:
            writer.writerow(board + [winner, ply])

    print(f"✅ Đã ghi file CSV: {output_path}")
    print(f"Tổng số bản ghi: {len(records)}")

if __name__ == "__main__":
    # --- Sửa đường dẫn ở đây ---
    input_file = "tictactoe4x4_dataset.bin"
    output_file = "tictactoe4x4_dataset.csv"
    convert_bin_to_csv(input_file, output_file)
