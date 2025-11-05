def generate_base_masks(n):
    """
    Tạo base masks tối thiểu cho bảng NxN.
    - Nếu n chẵn: num_rows = n // 2
    - Nếu n lẻ: num_rows = n // 2 + 1 (vì có hàng giữa)
    """
    size = n * n
    masks = []
    
    # Số hàng ngang cần
    if n % 2 == 0:  # n chẵn
        num_rows = n // 2
    else:  # n lẻ
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

def print_board(mask, n):
    """In board dạng ma trận NxN"""
    for i in range(n):
        row = mask[i*n:(i+1)*n]
        print("  " + " ".join(str(x) for x in row))

print("="*60)
print("BASE MASKS FOR 4x4 TIC TAC TOE")
print("="*60)
print(f"n = 4 (chẵn) → num_rows = 4 // 2 = 2 hàng")
print(f"Total: 3 cases (2 hàng + 1 chéo) × 2 players = 6 base masks\n")

masks_4x4 = generate_base_masks(4)

for i, mask in enumerate(masks_4x4, 1):
    player = 1 if mask[0] == 1 or (mask[0] == 0 and 1 in mask) else 2
    if i <= 3:
        if i == 1:
            print(f"Mask {i} - Player {player} - Hàng 0 (top):")
        elif i == 2:
            print(f"Mask {i} - Player {player} - Hàng 1:")
        else:
            print(f"Mask {i} - Player {player} - Đường chéo chính:")
    else:
        if i == 4:
            print(f"Mask {i} - Player {player} - Hàng 0 (top):")
        elif i == 5:
            print(f"Mask {i} - Player {player} - Hàng 1:")
        else:
            print(f"Mask {i} - Player {player} - Đường chéo chính:")
    print_board(mask, 4)
    print()

print("CSV FORMAT (4x4):")
print("-" * 60)
for mask in masks_4x4:
    print(", ".join(map(str, mask)))

print("\n" + "="*60)
print("BASE MASKS FOR 5x5 TIC TAC TOE")
print("="*60)
print(f"n = 5 (lẻ) → num_rows = 5 // 2 + 1 = 3 hàng (có hàng giữa)")
print(f"Total: 4 cases (3 hàng + 1 chéo) × 2 players = 8 base masks\n")

masks_5x5 = generate_base_masks(5)

for i, mask in enumerate(masks_5x5, 1):
    player = 1 if mask[0] == 1 or (mask[0] == 0 and 1 in mask) else 2
    if i <= 4:
        if i == 1:
            print(f"Mask {i} - Player {player} - Hàng 0 (top):")
        elif i == 2:
            print(f"Mask {i} - Player {player} - Hàng 1:")
        elif i == 3:
            print(f"Mask {i} - Player {player} - Hàng 2 (middle):")
        else:
            print(f"Mask {i} - Player {player} - Đường chéo chính:")
    else:
        if i == 5:
            print(f"Mask {i} - Player {player} - Hàng 0 (top):")
        elif i == 6:
            print(f"Mask {i} - Player {player} - Hàng 1:")
        elif i == 7:
            print(f"Mask {i} - Player {player} - Hàng 2 (middle):")
        else:
            print(f"Mask {i} - Player {player} - Đường chéo chính:")
    print_board(mask, 5)
    print()

print("CSV FORMAT (5x5):")
print("-" * 60)
for mask in masks_5x5:
    print(", ".join(map(str, mask)))

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print("4x4 (chẵn): 2 hàng + 1 chéo = 3 cases × 2 players = 6 base masks")
print("5x5 (lẻ):   3 hàng + 1 chéo = 4 cases × 2 players = 8 base masks")
print("\nVia D8 symmetry:")
print("  - 4x4: 6 base masks → expand to 12 unique win lines")
print("  - 5x5: 8 base masks → expand to 12 unique win lines")