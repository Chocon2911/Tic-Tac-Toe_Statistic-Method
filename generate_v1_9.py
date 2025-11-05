#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tic-Tac-Toe terminal dataset generator (wins only, no draws) for N×N (N >= 3)
- Bitboard representation for speed (two int masks: X, O)
- Canonicalization under D4 symmetries using precomputed permutations
- Streaming CSV writer (no giant in-memory rows list)
- Optional parallelization by line via --jobs (default: 1 = single-process)

CSV columns: board (compact "-XO" string of canonical board), winner (1=X,2=O), ply, N
"""

import argparse
import csv
import itertools
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

# =========================
# CONFIG (defaults in file)
# =========================
DEFAULT_N = 4
DEFAULT_OUT = "tictactoe.csv"
DEFAULT_JOBS = 1  # set >1 to enable multiprocessing by win-line

# =========================
# Helpers
# =========================

def popcount(x: int) -> int:
    return x.bit_count()


def indices_from_mask(mask: int, size: int) -> List[int]:
    out = []
    i = 0
    while mask:
        lsb = mask & -mask
        idx = (lsb.bit_length() - 1)
        out.append(idx)
        mask ^= lsb
        i += 1
    return out


def mask_from_indices(indices: Sequence[int]) -> int:
    m = 0
    for i in indices:
        m |= (1 << i)
    return m


# =========================
# Core context per board N
# =========================

@dataclass
class TTContext:
    N: int

    def __post_init__(self):
        assert self.N >= 3, "N phải >= 3"
        self.SIZE = self.N * self.N
        # index helpers
        def rc_to_i(r: int, c: int) -> int:
            return r * self.N + c

        def i_to_rc(i: int) -> Tuple[int, int]:
            return divmod(i, self.N)

        self.rc_to_i = rc_to_i
        self.i_to_rc = i_to_rc

        # D4 transforms capturing N
        def t_identity(r, c): return (r, c)
        def t_rot90(r, c): return (c, self.N - 1 - r)
        def t_rot180(r, c): return (self.N - 1 - r, self.N - 1 - c)
        def t_rot270(r, c): return (self.N - 1 - c, r)
        def t_reflect_h(r, c): return (self.N - 1 - r, c)
        def t_reflect_v(r, c): return (r, self.N - 1 - c)
        def t_reflect_main(r, c): return (c, r)
        def t_reflect_anti(r, c): return (self.N - 1 - c, self.N - 1 - r)

        self.TRANSFORMS = [
            t_identity, t_rot90, t_rot180, t_rot270,
            t_reflect_h, t_reflect_v, t_reflect_main, t_reflect_anti,
        ]
        # PERMS: perm[new_idx] = old_idx
        self.PERMS: List[List[int]] = [self._build_perm(T) for T in self.TRANSFORMS]
        # win masks (rows, cols, diags)
        self.WIN_MASKS: List[int] = self._generate_win_masks()
        # per-line remaining index list (to avoid recomputation)
        all_indices = tuple(range(self.SIZE))
        self.LINE_REMAINING: List[Tuple[int, ...]] = []
        for wm in self.WIN_MASKS:
            line_idxs = set(indices_from_mask(wm, self.SIZE))
            rem = tuple(i for i in all_indices if i not in line_idxs)
            self.LINE_REMAINING.append(rem)
        # ply bounds
        self.MIN_X_PLY = 2 * self.N - 1
        self.MIN_O_PLY = 2 * self.N
        self.MAX_PLY = self.SIZE

    # ---- internal builders ----
    def _build_perm(self, T) -> List[int]:
        perm = [0] * self.SIZE
        for old in range(self.SIZE):
            r, c = self.i_to_rc(old)
            r2, c2 = T(r, c)
            new = self.rc_to_i(r2, c2)
            perm[new] = old
        return perm

    def _generate_win_masks(self) -> List[int]:
        N = self.N
        rc_to_i = self.rc_to_i
        masks = []
        # rows
        for r in range(N):
            m = 0
            for c in range(N):
                m |= 1 << rc_to_i(r, c)
            masks.append(m)
        # cols
        for c in range(N):
            m = 0
            for r in range(N):
                m |= 1 << rc_to_i(r, c)
            masks.append(m)
        # main diag
        m = 0
        for i in range(N):
            m |= 1 << rc_to_i(i, i)
        masks.append(m)
        # anti diag
        m = 0
        for i in range(N):
            m |= 1 << rc_to_i(i, N - 1 - i)
        masks.append(m)
        return masks

    # ---- board ops ----
    def apply_perm_mask(self, mask: int, perm: Sequence[int]) -> int:
        # perm[new] = old
        out = 0
        for new in range(self.SIZE):
            old = perm[new]
            if (mask >> old) & 1:
                out |= (1 << new)
        return out

    def canonical_pair(self, x_mask: int, o_mask: int) -> Tuple[int, int]:
        best = None
        for perm in self.PERMS:
            x2 = self.apply_perm_mask(x_mask, perm)
            o2 = self.apply_perm_mask(o_mask, perm)
            t = (x2, o2)
            if best is None or t < best:
                best = t
        return best  # type: ignore

    def has_winner_mask(self, mask: int) -> bool:
        for wm in self.WIN_MASKS:
            if (mask & wm) == wm:
                return True
        return False

    def winners(self, x_mask: int, o_mask: int) -> Tuple[bool, bool]:
        return self.has_winner_mask(x_mask), self.has_winner_mask(o_mask)

    def valid_turn_counts(self, x_mask: int, o_mask: int) -> bool:
        x = popcount(x_mask)
        o = popcount(o_mask)
        if not (x == o or x == o + 1):
            return False
        xw, ow = self.winners(x_mask, o_mask)
        if ow and x != o:
            return False
        if xw and x != o + 1:
            return False
        return True

    def min_ply_ok(self, x_mask: int, o_mask: int) -> bool:
        x = popcount(x_mask)
        o = popcount(o_mask)
        if self.has_winner_mask(x_mask) and (2 * x - 1) < self.MIN_X_PLY:
            return False
        if self.has_winner_mask(o_mask) and (2 * o) < self.MIN_O_PLY:
            return False
        return True

    def is_terminal_reachable(self, x_mask: int, o_mask: int, winner: int) -> bool:
        xw, ow = self.winners(x_mask, o_mask)
        if not ((winner == 1 and xw and not ow) or (winner == 2 and ow and not xw)):
            return False
        x = popcount(x_mask)
        o = popcount(o_mask)
        if winner == 1:
            if x != o + 1:
                return False
            # remove one X on a winning line and previous must be valid non-win
            for wm in self.WIN_MASKS:
                if (x_mask & wm) == wm:
                    # iterate bits in wm
                    m = wm
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        prev_x = x_mask & ~(1 << idx)
                        prev_o = o_mask
                        if popcount(prev_x) == x - 1 and not any(self.winners(prev_x, prev_o)) and self.valid_turn_counts(prev_x, prev_o):
                            return True
                        m ^= lsb
        else:  # winner == 2
            if x != o:
                return False
            for wm in self.WIN_MASKS:
                if (o_mask & wm) == wm:
                    m = wm
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        prev_o = o_mask & ~(1 << idx)
                        prev_x = x_mask
                        if popcount(prev_o) == o - 1 and not any(self.winners(prev_x, prev_o)) and self.valid_turn_counts(prev_x, prev_o):
                            return True
                        m ^= lsb
        return False


# =========================
# Enumeration (wins only)
# =========================

def enumerate_wins_for_line(ctx: TTContext, line_mask: int, rem_indices: Tuple[int, ...], winner: int) -> Iterable[Tuple[int, int]]:
    """Yield (x_mask, o_mask) for terminal wins where `winner` wins on this `line_mask`."""
    N = ctx.N
    SIZE = ctx.SIZE

    if winner == 1:
        ply_values = range(ctx.MIN_X_PLY, ctx.MAX_PLY + 1, 2)  # odd
    else:
        ply_values = range(ctx.MIN_O_PLY, ctx.MAX_PLY + 1, 2)  # even

    for T in ply_values:
        if winner == 1:
            x_total = (T + 1) // 2
            o_total = (T - 1) // 2
            extra_w = x_total - N
            extra_other = o_total
            w_is_x = True
        else:
            x_total = T // 2
            o_total = T // 2
            extra_w = o_total - N
            extra_other = x_total
            w_is_x = False

        if extra_w < 0 or extra_other < 0:
            continue
        if extra_w + extra_other > len(rem_indices):
            continue

        # choose cells for extra winner stones outside the winning line
        for extra_w_cells in itertools.combinations(rem_indices, extra_w):
            extra_w_set = set(extra_w_cells)
            remaining_for_other = [i for i in rem_indices if i not in extra_w_set]
            if extra_other > len(remaining_for_other):
                continue

            # choose opponent cells
            for other_cells in itertools.combinations(remaining_for_other, extra_other):
                # construct masks
                ew_mask = mask_from_indices(extra_w_cells)
                other_mask = mask_from_indices(other_cells)
                if w_is_x:
                    x_mask = line_mask | ew_mask
                    o_mask = other_mask
                else:
                    o_mask = line_mask | ew_mask
                    x_mask = other_mask

                # sanity: disjoint masks
                if (x_mask & o_mask) != 0:
                    continue

                # validations
                xw, ow = ctx.winners(x_mask, o_mask)
                if not ((winner == 1 and xw and not ow) or (winner == 2 and ow and not xw)):
                    continue
                if not ctx.valid_turn_counts(x_mask, o_mask):
                    continue
                if not ctx.min_ply_ok(x_mask, o_mask):
                    continue
                if not ctx.is_terminal_reachable(x_mask, o_mask, winner):
                    continue

                yield (x_mask, o_mask)


# =========================
# CSV I/O
# =========================

def board_to_compact_from_masks(size: int, x_mask: int, o_mask: int) -> str:
    chars = []
    for i in range(size):
        if (x_mask >> i) & 1:
            chars.append('X')
        elif (o_mask >> i) & 1:
            chars.append('O')
        else:
            chars.append('-')
    return ''.join(chars)


# =========================
# Core generation
# =========================

def generate_dataset(N: int, out_path: str, jobs: int = 1) -> None:
    t0 = time.time()
    ctx = TTContext(N)

    # Prepare CSV
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    f = open(out_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["board", "winner", "ply", "N"])

    print("=" * 60)
    print(f"{N}x{N} TIC-TAC-TOE DATASET GENERATOR (wins only, canonical symmetry)")
    print("=" * 60)

    seen = set()  # keys: (canonical_x, canonical_o, winner)
    total_written = 0

    # iterate each win line
    print("\n[1/1] Generating terminal wins (no draws)...")
    win_start = time.time()

    tasks = [(li, line_mask, ctx.LINE_REMAINING[li], winner)
             for li, line_mask in enumerate(ctx.WIN_MASKS)
             for winner in (1, 2)]

    if jobs <= 1:
        for li, line_mask, rem, winner in tasks:
            line_count = 0
            for x_mask, o_mask in enumerate_wins_for_line(ctx, line_mask, rem, winner):
                cx, co = ctx.canonical_pair(x_mask, o_mask)
                key = (cx, co, winner)
                if key in seen:
                    continue
                seen.add(key)
                ply = popcount(cx | co)
                w.writerow([board_to_compact_from_masks(ctx.SIZE, cx, co), winner, ply, N])
                line_count += 1
                total_written += 1
            print(f"  Line {li+1}/{len(ctx.WIN_MASKS)}, winner={winner}: +{line_count} (acc={total_written})")
    else:
        # optional: multiprocessing over lines (coarse). Note: results buffered per task.
        from multiprocessing import Pool

        def _worker(args):
            li, line_mask, rem, winner, N_local = args
            c = TTContext(N_local)
            out = []
            for x_mask, o_mask in enumerate_wins_for_line(c, line_mask, rem, winner):
                cx, co = c.canonical_pair(x_mask, o_mask)
                ply = popcount(cx | co)
                out.append((cx, co, winner, ply))
            return li, winner, out

        with Pool(processes=jobs) as pool:
            for li, winner, out_list in pool.imap_unordered(
                _worker,
                [(li, lm, rem, winner, N) for (li, lm, rem, winner) in tasks],
                chunksize=1,
            ):
                line_count = 0
                for cx, co, winner, ply in out_list:
                    key = (cx, co, winner)
                    if key in seen:
                        continue
                    seen.add(key)
                    w.writerow([board_to_compact_from_masks(ctx.SIZE, cx, co), winner, ply, N])
                    line_count += 1
                    total_written += 1
                print(f"  Line {li+1}/{len(ctx.WIN_MASKS)}, winner={winner}: +{line_count} (acc={total_written})")

    f.flush()
    f.close()

    print(f"  ✓ Wins done in {time.time() - win_start:.2f}s")

    # summary
    dt = time.time() - t0
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total unique canonical boards: {total_written:,}")
    print("Include draws: no (disabled)")
    print(f"Output: {out_path}")
    print(f"Total time: {dt:.2f}s")
    print("=" * 60 + "\n")


# =========================
# CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="Generate terminal Tic-Tac-Toe datasets for NxN (N>=3). Wins only.")
    ap.add_argument("--n", type=int, help=f"Board size N (>=3). If omitted, uses DEFAULT_N={DEFAULT_N}.")
    ap.add_argument("--out", type=str, help=f"Output CSV path. If omitted, uses DEFAULT_OUT='{DEFAULT_OUT}'.")
    ap.add_argument("--jobs", type=int, default=None, help=f"Parallel workers (by line). Default in-file = {DEFAULT_JOBS}.")
    args = ap.parse_args()

    N = args.n if args.n is not None else DEFAULT_N
    out_path = args.out if args.out is not None else DEFAULT_OUT
    jobs = args.jobs if args.jobs is not None else DEFAULT_JOBS

    print(f"[config] N={N}, out='{out_path}', jobs={jobs}")
    generate_dataset(N=N, out_path=out_path, jobs=jobs)


if __name__ == "__main__":
    main()
