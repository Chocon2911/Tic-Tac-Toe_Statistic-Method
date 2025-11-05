#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import time
from typing import List, Tuple, Set, Iterable
import csv

# =========================
# CONFIG (defaults in file)
# =========================
DEFAULT_N = 4                         # 3 / 4 / 5 / ...
DEFAULT_OUT = "tictactoe.csv"         # output CSV file path
DEFAULT_INCLUDE_DRAWS = False         # True to also generate draws
DRAW_MAX_PATTERNS = 2_000_000         # skip draws if 2^SIZE > this

# =========================
# Core context per board N
# =========================

class TTContext:
    def __init__(self, N: int):
        assert N >= 3, "N phải >= 3"
        self.N = N
        self.SIZE = N * N
        self.ALL = tuple(range(self.SIZE))

        # index helpers
        def rc_to_i(r: int, c: int) -> int: return r * N + c
        def i_to_rc(i: int) -> Tuple[int, int]: return divmod(i, N)
        self.rc_to_i = rc_to_i
        self.i_to_rc = i_to_rc

        # build D4 transforms (capture N)
        def t_identity(r,c):      return (r, c)
        def t_rot90(r,c):         return (c, N-1-r)
        def t_rot180(r,c):        return (N-1-r, N-1-c)
        def t_rot270(r,c):        return (N-1-c, r)
        def t_reflect_h(r,c):     return (N-1-r, c)
        def t_reflect_v(r,c):     return (r, N-1-c)
        def t_reflect_main(r,c):  return (c, r)
        def t_reflect_anti(r,c):  return (N-1-c, N-1-r)

        self.TRANSFORMS = [
            t_identity, t_rot90, t_rot180, t_rot270,
            t_reflect_h, t_reflect_v, t_reflect_main, t_reflect_anti
        ]

        # precompute perms for fast canonical (perm[new_idx] = old_idx)
        self.PERMS = [self._build_perm(T) for T in self.TRANSFORMS]

        # all winning lines: rows, cols, 2 diagonals (length N)
        self.WIN_LINES = self._generate_win_lines()

        # ply bounds
        self.MIN_X_PLY = 2 * N - 1   # X có thể thắng lần đầu
        self.MIN_O_PLY = 2 * N       # O có thể thắng lần đầu
        self.MAX_PLY   = self.SIZE

    def _build_perm(self, T):
        perm = [0] * self.SIZE
        for old in range(self.SIZE):
            r, c = self.i_to_rc(old)
            r2, c2 = T(r, c)
            new = self.rc_to_i(r2, c2)
            perm[new] = old
        return perm

    def canonical_board(self, board: List[int]) -> Tuple[int, ...]:
        # fastest: use precomputed perms
        best = None
        for perm in self.PERMS:
            out = tuple(board[perm[k]] for k in range(self.SIZE))
            if best is None or out < best:
                best = out
        return best

    def _generate_win_lines(self) -> List[Tuple[int,...]]:
        N = self.N
        rc_to_i = self.rc_to_i
        lines = []
        for r in range(N):
            lines.append(tuple(rc_to_i(r, c) for c in range(N)))  # rows
        for c in range(N):
            lines.append(tuple(rc_to_i(r, c) for r in range(N)))  # cols
        lines.append(tuple(rc_to_i(i, i) for i in range(N)))      # main diag
        lines.append(tuple(rc_to_i(i, N-1-i) for i in range(N)))  # anti diag
        return lines

    # -------------- game checks --------------

    def has_winner(self, board: List[int], player: int) -> bool:
        for line in self.WIN_LINES:
            ok = True
            for i in line:
                if board[i] != player:
                    ok = False
                    break
            if ok:
                return True
        return False

    def winners(self, board: List[int]) -> Set[int]:
        s = set()
        if self.has_winner(board, 1): s.add(1)
        if self.has_winner(board, 2): s.add(2)
        return s

    def count_pieces(self, board: List[int]) -> Tuple[int,int]:
        x = 0; o = 0
        for v in board:
            if v == 1: x += 1
            elif v == 2: o += 1
        return x, o

    def valid_turn_counts(self, board: List[int]) -> bool:
        x, o = self.count_pieces(board)
        if not (x == o or x == o + 1):   # X luôn đi trước
            return False
        if self.has_winner(board, 2) and x != o:       # O thắng → ply chẵn
            return False
        if self.has_winner(board, 1) and x != o + 1:   # X thắng → ply lẻ
            return False
        return True

    def min_ply_ok(self, board: List[int]) -> bool:
        x, o = self.count_pieces(board)
        if self.has_winner(board, 1) and (2*x - 1) < self.MIN_X_PLY:
            return False
        if self.has_winner(board, 2) and (2*o) < self.MIN_O_PLY:
            return False
        return True

    def is_terminal_reachable(self, board: List[int], winner: int) -> bool:
        """Đúng 'nước cuối': bỏ 1 quân thắng trên 1 line thắng → bàn trước hợp lệ & chưa ai thắng."""
        if self.winners(board) != {winner}:
            return False

        x, o = self.count_pieces(board)
        if winner == 1:
            if x != o + 1: return False
            prev_x, prev_o = x - 1, o
        else:
            if x != o: return False
            prev_x, prev_o = x, o - 1

        for line in self.WIN_LINES:
            ok_line = True
            for i in line:
                if board[i] != winner:
                    ok_line = False
                    break
            if not ok_line: 
                continue
            for i in line:
                if board[i] != winner:
                    continue
                prev = board.copy()
                prev[i] = 0
                if self.count_pieces(prev) != (prev_x, prev_o):
                    continue
                if self.winners(prev):    # bàn trước vẫn thắng → không phải nước cuối
                    continue
                if not self.valid_turn_counts(prev):
                    continue
                return True
        return False

# =========================
# Enumerations (wins/draws)
# =========================

def enumerate_wins_for_line(ctx: TTContext, line: Tuple[int,...], winner: int) -> Iterable[List[int]]:
    """Sinh mọi bàn THẮNG hợp lệ, terminal, người 'winner' thắng trên 'line'."""
    N = ctx.N
    SIZE = ctx.SIZE
    ALL = ctx.ALL

    if winner == 1:
        ply_values = range(ctx.MIN_X_PLY, ctx.MAX_PLY + 1, 2)  # odd
    else:
        ply_values = range(ctx.MIN_O_PLY, ctx.MAX_PLY + 1, 2)  # even

    remaining = tuple(i for i in ALL if i not in line)

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
        if extra_w + extra_other > len(remaining):
            continue

        for extra_w_cells in itertools.combinations(remaining, extra_w):
            rem2 = [i for i in remaining if i not in extra_w_cells]
            if extra_other > len(rem2): 
                continue
            for other_cells in itertools.combinations(rem2, extra_other):
                b = [0]*SIZE
                # fill winning line
                for i in line: b[i] = w_val
                # fill extra winner stones outside the line
                for i in extra_w_cells: b[i] = w_val
                # check conflict then place opponent
                if any(b[i] != 0 for i in other_cells): 
                    continue
                for i in other_cells: b[i] = other_val

                # validations
                if ctx.winners(b) != {winner}: 
                    continue
                if not ctx.valid_turn_counts(b): 
                    continue
                if not ctx.min_ply_ok(b): 
                    continue
                if not ctx.is_terminal_reachable(b, winner): 
                    continue

                yield b

def enumerate_draws_fullboard(ctx: TTContext) -> Iterable[List[int]]:
    """Hòa: bàn đầy, không ai thắng, lượt đếm hợp lệ. 
       Bỏ qua nếu 2^SIZE > DRAW_MAX_PATTERNS.
    """
    SIZE = ctx.SIZE
    if (1 << SIZE) > DRAW_MAX_PATTERNS:
        return  # skip (quá lớn)

    for pat in itertools.product((1,2), repeat=SIZE):
        b = list(pat)
        if ctx.has_winner(b, 1) or ctx.has_winner(b, 2):
            continue
        x, o = ctx.count_pieces(b)
        if x == o or x == o + 1:
            yield b

# =========================
# CSV I/O
# =========================

def board_to_compact(board: List[int]) -> str:
    return ''.join('-' if v == 0 else 'X' if v == 1 else 'O' for v in board)

def write_compact_csv(path: str, rows: List[Tuple[str, int, int, int]]):
    """rows: (board_str, winner, ply, N)"""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["board", "winner", "ply", "N"])
        w.writerows(rows)

# =========================
# Main driver
# =========================

def generate_dataset(N: int,
                     out_path: str,
                     include_draws: bool = False) -> None:
    t0 = time.time()
    ctx = TTContext(N)

    print("="*60)
    print(f"{N}x{N} TIC-TAC-TOE DATASET GENERATOR (canonical symmetry)")
    print("="*60)

    # ---- generate wins ----
    rows: List[Tuple[str,int,int,int]] = []
    seen = set()  # (canonical_board_tuple, winner)

    print("\n[1/2] Generating terminal wins...")
    win_start = time.time()
    for li, line in enumerate(ctx.WIN_LINES, 1):
        line_count = 0
        for winner in (1, 2):
            for b in enumerate_wins_for_line(ctx, line, winner):
                cb = ctx.canonical_board(b)
                key = (cb, winner)
                if key in seen:
                    continue
                seen.add(key)
                rows.append((board_to_compact(cb), winner, sum(1 for v in cb if v != 0), N))
                line_count += 1
        print(f"  Line {li}/{len(ctx.WIN_LINES)}: +{line_count} boards (acc={len(rows)})")
    print(f"  ✓ Wins done in {time.time()-win_start:.2f}s")

    # ---- draws (optional) ----
    if include_draws:
        print("\n[2/2] Generating draws (full-board, no winner)...")
        draw_start = time.time()
        added = 0
        for b in enumerate_draws_fullboard(ctx) or []:
            cb = ctx.canonical_board(b)
            key = (cb, 0)
            if key in seen:
                continue
            seen.add(key)
            rows.append((board_to_compact(cb), 0, ctx.SIZE, N))
            added += 1
        print(f"  ✓ Draws +{added} in {time.time()-draw_start:.2f}s")
    else:
        print("\n[2/2] Skipped draws (use --include-draws to enable).")

    # ---- write CSV ----
    print("\nWriting CSV...")
    write_compact_csv(out_path, rows)
    dt = time.time() - t0

    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"Total unique canonical boards: {len(rows):,}")
    print(f"Include draws: {'yes' if include_draws else 'no'}")
    print(f"Output: {out_path}")
    print(f"Total time: {dt:.2f}s")
    print("="*60 + "\n")

# =========================
# CLI (with in-file defaults)
# =========================

def main():
    ap = argparse.ArgumentParser(description="Generate terminal Tic-Tac-Toe datasets for NxN (N>=3).")
    ap.add_argument("--n", type=int, help=f"Board size N (>=3). If omitted, uses DEFAULT_N={DEFAULT_N}.")
    ap.add_argument("--out", type=str, help=f"Output CSV path. If omitted, uses DEFAULT_OUT='{DEFAULT_OUT}'.")
    ap.add_argument("--include-draws", action="store_true", help=f"Also generate draws (full board). Default in-file = {DEFAULT_INCLUDE_DRAWS}.")
    args = ap.parse_args()

    N = args.n if args.n is not None else DEFAULT_N
    out_path = args.out if args.out is not None else DEFAULT_OUT
    include_draws = args.include_draws or DEFAULT_INCLUDE_DRAWS

    print(f"[config] N={N}, out='{out_path}', include_draws={include_draws}, DRAW_MAX_PATTERNS={DRAW_MAX_PATTERNS}")
    generate_dataset(N=N, out_path=out_path, include_draws=include_draws)

if __name__ == "__main__":
    main()
