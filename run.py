# ==== DEFAULTS WHEN NO CLI ARGS ====
DEFAULT_N = 3                   # đổi tùy ý: 3 / 4 / 5 / ...
DEFAULT_OUT = "tictactoe.csv"     # đường dẫn file xuất CSV
DEFAULT_INCLUDE_DRAWS = False     # True nếu muốn sinh cả hòa (N lớn sẽ chậm)

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Generate terminal Tic-Tac-Toe datasets for NxN (N>=3).")
    ap.add_argument("--n", type=int, help="Board size N (>=3). If omitted, uses DEFAULT_N in file.")
    ap.add_argument("--out", type=str, help="Output CSV path. If omitted, uses DEFAULT_OUT in file.")
    ap.add_argument("--include-draws", action="store_true", help="Also generate draws (full board).")
    args = ap.parse_args()

    # Nếu tham số không được truyền, dùng mặc định trong file
    N = args.n if args.n is not None else DEFAULT_N
    out_path = args.out if args.out is not None else DEFAULT_OUT
    include_draws = args.include_draws or DEFAULT_INCLUDE_DRAWS

    generate_dataset(N=N, out_path=out_path, include_draws=include_draws)

if __name__ == "__main__":
    import sys
    # Không có tham số → chạy với DEFAULT_*; có tham số → cho main() xử lý
    if len(sys.argv) == 1:
        print("[i] No CLI args detected → using defaults in file.")
        generate_dataset(N=DEFAULT_N, out_path=DEFAULT_OUT, include_draws=DEFAULT_INCLUDE_DRAWS)
    else:
        main()
