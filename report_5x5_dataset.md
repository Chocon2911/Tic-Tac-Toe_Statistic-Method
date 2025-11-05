### 1. Introduction

This document describes how we generate a comprehensive dataset of terminal, legal Tic‑Tac‑Toe positions on a 5×5 board (win condition: exactly five‑in‑a‑row). The approach is engineered for correctness, deduplication under board symmetries, and scalability.

#### 1.1 Goals
- Capture every terminal position that could appear at the end of a legal game starting from an empty board, with X moving first.
- Ensure exactly one winner exists in each position, and the winning line is formed on the last move.
- Remove duplicates created by geometric symmetries of the square board.
- Encode results compactly and stream them to disk by ply to control memory usage.

#### 1.2 Scope and constraints
- Board size: strictly 5×5 (25 cells). Players: X=1 and O=2; empty=0.
- Dataset coverage: terminal wins only — positions where either X or O has exactly five‑in‑a‑row and the game just ended on the last move.
- Explicitly excluded: mid‑game/ongoing states, non‑terminal states, and draws (no‑winner full boards).
- Legal turn order: counts must satisfy X starts, so `x == o` or `x == o + 1`, consistent with who wins.
- Minimum feasible plies for 5×5: X can first win at 9 plies (2·5−1), O at 10 plies (2·5).
- Computation targets completeness of terminal wins while remaining practical via symmetry and streaming.

#### 1.3 Outputs
- Layered binary files, one per ply (e.g., `layer_06.bin` … `layer_25.bin`).
- Each record stores a deduplicated, canonical board encoding of a terminal win.


### 2. Reverse Generation

Instead of exploring the forward game tree (which is combinatorially explosive), we construct terminal states directly by working backwards from a known winning line.

#### 2.1 High‑level idea (terminal wins only)
1) Pick a geometric winning line (5 contiguous cells). 2) Choose the winner and a feasible total ply `T` (odd for X, even for O) within 5×5 ranges. 3) Place the five winning marks on that line. 4) Distribute the remaining marks off the line to satisfy piece counts and legality. 5) Validate the final board as a true terminal state with a plausible last move. Non‑terminal and draw positions are not considered.

#### 2.2 Ply and piece accounting
- If X wins on ply `T` (odd): `x_total = (T + 1) / 2`, `o_total = (T − 1) / 2`.
- If O wins on ply `T` (even): `x_total = T / 2`, `o_total = T / 2`.
- Off‑line allocation for the winner: `extra_w = (winner_total − N)`; opponent places all of their pieces off the line.

Typical feasible ranges for 5×5 (further filtered by legality):
- X wins: odd `T` in {9, 11, 13, …, 25} (minimum derived from `2N−1`).
- O wins: even `T` in {10, 12, 14, …, 24} (minimum derived from `2N`).

#### 2.3 Enumeration procedure (per line, per winner)
1) Precompute the set of cells not on the line (20 cells).
2) Choose `extra_w` winner cells among those 20.
3) From the remaining cells, choose `extra_other` opponent cells.
4) Construct the board (line = winner, extras off‑line placed, no overlaps).
5) Run legality checks (Section 2.4). If all pass, keep the board.

#### 2.4 Legality checks (strict)
- Single‑winner condition: Exactly one player must have a five‑in‑a‑row.
- Turn counts: `x == o` or `x == o + 1`; moreover, if O wins then `x == o`, if X wins then `x == o + 1`.
- Minimum ply feasibility for 5‑in‑a‑row on 5×5:
  - X wins only at or after X’s N‑th move → total plies ≥ `2N−1` (for 5×5: 9).
  - O wins only at or after O’s N‑th move → total plies ≥ `2N` (for 5×5: 10).
- Terminal reachability (last‑move test): There exists a cell on some winning line where removing that mark yields a position with correct counts, no winner, and valid turn parity. This models a legal preceding position and ensures the current position can arise as a last move.

#### 2.5 Why this is efficient
- Massive pruning versus naive search: we only examine boards that already contain a complete five‑in‑a‑row and are consistent with counts.
- Independence per line and ply: makes the computation easily streamable and parallelizable.


### 3. Symmetrical Optimization

We leverage the D4 symmetry group of the square: 4 rotations (0°, 90°, 180°, 270°) and 4 reflections (horizontal, vertical, main diagonal, anti‑diagonal).

#### 3.1 Expanding winning lines via symmetry
- Start from a compact set of 5×5 base line masks generated automatically: 3 horizontal rows (including middle row) + 1 main diagonal, duplicated for both players → 8 base masks.
- Apply all 8 transforms to each mask to generate all distinct geometric winning lines.
- Collect uniqueness under geometry; for 5×5, this expands to 12 unique winning lines (per visualization output).

Benefits: minimal input specification, guaranteed coverage of all line orientations and positions.

#### 3.2 Canonicalization of boards
- For each generated board, compute all 8 symmetric variants.
- Select the lexicographically smallest variant as the canonical representative.
- Maintain a `seen` set of canonical boards. If a new board’s canonical form is already present, discard it.

Benefits: eliminates duplicates across different lines or generation paths; reduces I/O and storage without losing information.


### 4. Encoding dataset

The dataset is streamed to disk using a compact, fixed‑width binary encoding for 5×5.

#### 4.1 Board encoding
- 2 bits per cell: `00=empty`, `01=X`, `10=O`.
- 25 cells → 50 bits; packed into a single 64‑bit unsigned little‑endian word for simplicity and alignment.

- Record: 8 bytes → the 64‑bit packed board only (winner and ply are implicit from layer/win detection if not stored separately).
- File header (per layer file): 8 bytes total
  - 4 bytes: magic number `0x54545435` ("TTT5").
  - 4 bytes: record count (written as 0 initially and finalized after the layer completes).

#### 4.3 Layering strategy
- One file per ply: `layer_09.bin` … `layer_25.bin` (5×5). Only terminal win layers are generated; draws and ongoing states are not written.
- Advantages:
  - Streaming: no need to retain entire result sets in memory.
  - Targeted analysis: load only specific game lengths.
  - Parallelism: different layers can be generated or consumed independently.

#### 4.4 Streaming workflow
1) Initialize the layer file and write the header placeholder. Output directory: `dataset_5x5_final`.
2) For each validated canonical board, append its 8‑byte record immediately.
3) After finishing the layer, seek back and finalize the record count in the header.

#### 4.5 Integrity considerations
- The canonicalization set lives in memory for deduplication across the run.
- The final record count is computed from file size to avoid mismatch.
- Deterministic enumeration and stable canonicalization guarantee reproducibility.


### 5. Conclusion

The generation pipeline unifies reverse construction from winning lines, rigorous legality validation, symmetry‑aware expansion and canonical deduplication, plus a compact streaming binary format. This yields a complete, duplicate‑free corpus of terminal 5×5 Tic‑Tac‑Toe positions that is scalable to produce and efficient to store and analyze. Future enhancements could include explicit metadata per record (e.g., storing winner/ply alongside the board), multi‑threaded enumeration per layer, and optional CSV export for quick inspection.


### 6. Dataset examples (readable format)

To make the encoding and the terminal‑state definition concrete, below are small, human‑readable examples (not read from the binary files):

#### 6.1 Example A — X win on a horizontal line (ply = 9)
Board (X=1, O=2, −=empty):
```
X X X X X
− − − − −
− − O − −
− O − − −
− − − − −
```
- Winner: X (five in row 0)
- Counts: X=5, O=4 ⇒ total plies=9 (satisfies X starts: X = O + 1)
- Canonical form: The board is already minimal under D4 in this example.
- Encoding: 2 bits/cell packed in little‑endian cell order (row‑major from top‑left). The first 5 cells encode `01` repeatedly.

#### 6.2 Example B — O win on a diagonal (ply = 10)
```
O − − − −
− O − − −
− − O − X
− − − O −
− − − − O
```
- Winner: O (main diagonal)
- Counts: X=5, O=5 ⇒ total plies=10 (satisfies X starts: X = O)
- Encoding: winner is implicit in the board; file layering indicates total ply range.

Note: Actual dataset entries are deduplicated under symmetry. Variants of these positions that are rotations/reflections are represented only once via their canonical form. Only terminal wins are present; there are no draws or mid‑game examples in the dataset.
The visualization helper (`visual.py`) prints the base masks for 5×5 and summarizes how many unique winning lines are covered after applying D4 symmetry.


### 7. Dataset characteristics and quality assurance

#### 7.1 Cardinality trends by ply
- Lower plies (e.g., 6–8) are severely constrained; few or no reachable wins exist due to minimum‑ply feasibility.
- Mid plies see rapid growth as more off‑line pieces can be placed while maintaining single‑winner legality.
- Higher plies (approaching 25) eventually taper as the board fills and multi‑line winners or invalid counts are more common; canonical deduplication also reduces counts.

While exact counts depend on base lines and legality filters, the per‑layer files (`layer_09.bin` … `layer_25.bin`) naturally reflect this trend in their sizes. In practice for 5×5, layers begin at ply 9 by construction.

#### 7.2 Invariants enforced
- Single winner invariant: `winners(board) == {winner}`.
- Turn count invariant: `x == o` or `x == o + 1`; consistent with winner parity.
- Minimum‑ply invariant: X wins require ≥9 plies; O wins require ≥10 plies.
- Terminal reachability invariant: removing a last‑move cell on some winning line yields a legal, non‑winning prior position with correct counts.

#### 7.3 Deduplication soundness
- Canonicalization under all 8 D4 transforms ensures geometrically identical positions are counted once.
- The canonical representative is the lexicographically smallest image across transforms, providing a stable, deterministic key.


### 8. Performance and resource usage

#### 8.1 Time complexity considerations
- Enumeration scales with combinations of off‑line placements: `C(20, extra_w) × C(20−extra_w, extra_other)` per (line, winner, ply), pruned heavily by legality checks.
- Symmetry expansion of lines is O(#base_masks × 8), negligible relative to board enumeration.
- Canonicalization is O(8×SIZE) per candidate (apply 8 transforms and compare), cheap vs. combination enumeration.

#### 8.2 Practical runtime observations
- Auto‑computed minimum layer (`2N−1`) eliminates impossible early layers, reducing total work.
- Streaming per ply avoids holding all results in memory.
- Progress reporting per line/winner pair provides visibility into long‑running layers.
- Expected pattern: early feasible layers complete quickly (few valid boards), mid‑layers dominate runtime, late layers moderate again due to heavier pruning.

#### 8.3 Memory and I/O footprint
- In‑memory state: `seen` set for canonical boards plus current layer’s I/O buffers.
- On‑disk: each record is fixed at 8 bytes; header is 8 bytes per file. This allows estimating file sizes as `8 + 8 × (#records)` per layer.
- The 64‑bit aligned format improves write throughput on most systems.


### 9. Limitations and future work

#### 9.1 Limitations
- Winner/ply are not stored per record in the 5×5 binary format (they are derivable from layer and board content); downstream tasks that need explicit metadata must reconstruct it.
- Single‑threaded generation can be compute‑bound on mid‑layers.
- Canonicalization deduplicates geometric symmetries only; it does not deduplicate semantically equivalent histories (which is out of scope for terminal states).

#### 9.2 Future work
- Embed explicit per‑record metadata (winner, ply, line id) for direct consumption at slight storage cost.
- Parallelize enumeration across lines or layers; merge with thread‑safe canonical sets or sharded canonicalization.
- Provide an optional CSV mirror for small layers to facilitate quick human inspection.
- Add a validator tool that replays the last move automatically for random samples as a sanity check.


