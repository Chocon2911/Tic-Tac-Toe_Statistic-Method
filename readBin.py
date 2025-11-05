# Lưu
import struct

data = [10, 20, 30]
with open("tictactoe4x4_dataset.bin", "wb") as f:
    for num in data:
        f.write(struct.pack("i", num))  # i = int 4 byte
