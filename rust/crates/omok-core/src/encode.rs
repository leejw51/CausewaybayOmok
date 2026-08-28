//! Board -> network input planes.  Ported from `omok/encode.py`.
//!
//! Plane layout (NCHW, f32, always from the side-to-move's point of view):
//!
//! ```text
//! 0  stones of the player to move
//! 1  stones of the opponent
//! 2  one-hot of the opponent's last move
//! 3  one-hot of the player-to-move's previous move
//! 4  all ones when the player to move is black
//! ```

use crate::board::{other, Board, BLACK};

pub const NUM_PLANES: usize = 5;

/// Write one board's planes into `out`, which must be `NUM_PLANES * n * n` long.
pub fn encode_board(board: &Board, out: &mut [f32]) {
    let n = board.size;
    let area = n * n;
    debug_assert_eq!(out.len(), NUM_PLANES * area);
    out.fill(0.0);

    let me = board.to_move;
    let them = other(me);
    let (mine, theirs) = out[..2 * area].split_at_mut(area);
    for i in 0..area {
        mine[i] = (board.cells[i] == me) as u8 as f32;
        theirs[i] = (board.cells[i] == them) as u8 as f32;
    }
    let moves = &board.moves;
    if let Some(&last) = moves.last() {
        out[2 * area + last as usize] = 1.0;
    }
    if moves.len() > 1 {
        out[3 * area + moves[moves.len() - 2] as usize] = 1.0;
    }
    if me == BLACK {
        out[4 * area..5 * area].fill(1.0);
    }
}

/// Encode a batch of boards into one contiguous NCHW buffer.
pub fn encode_batch(boards: &[&Board]) -> Vec<f32> {
    let Some(first) = boards.first() else {
        return Vec::new();
    };
    let stride = NUM_PLANES * first.size * first.size;
    let mut out = vec![0.0f32; boards.len() * stride];
    for (i, board) in boards.iter().enumerate() {
        encode_board(board, &mut out[i * stride..(i + 1) * stride]);
    }
    out
}
