//! The rules the network was trained against, checked against the behaviour
//! `omok/board.py` specifies.

use omok_core::board::{Board, BLACK, EMPTY, WHITE};
use omok_core::encode::{encode_batch, encode_board, NUM_PLANES};
use omok_core::{format_move, parse_move};

fn board() -> Board {
    Board::new(15, 5, true)
}

#[test]
fn a_new_board_is_empty_with_black_to_move() {
    let b = board();
    assert_eq!(b.to_move, BLACK);
    assert_eq!(b.move_number(), 0);
    assert!(!b.over);
    assert!(b.cells.iter().all(|&c| c == EMPTY));
    assert_eq!(b.legal_moves().len(), 225);
}

#[test]
fn playing_alternates_colours() {
    let mut b = board();
    assert!(b.play(112));
    assert_eq!(b.cells[112], BLACK);
    assert_eq!(b.to_move, WHITE);
    assert!(b.play(113));
    assert_eq!(b.cells[113], WHITE);
    assert_eq!(b.to_move, BLACK);
}

#[test]
fn illegal_moves_are_rejected_without_changing_anything() {
    let mut b = board();
    b.play(112);
    assert!(!b.play(112), "cannot play on an occupied square");
    assert!(!b.play(225), "cannot play off the board");
    assert_eq!(b.move_number(), 1);
    assert_eq!(b.to_move, WHITE);
}

#[test]
fn five_in_a_row_wins_and_the_line_is_reported() {
    let mut b = board();
    for i in 0..5 {
        assert!(b.play(7 * 15 + i));
        if i < 4 {
            assert!(b.play(i));
        }
    }
    assert!(b.over);
    assert_eq!(b.winner, BLACK);
    let line = b.win_line();
    assert_eq!(line.len(), 5);
    assert!(line.iter().all(|&i| b.cells[i] == BLACK));
    // The side to move flips even on the winning move, and the loser reads -1.
    assert_eq!(b.to_move, WHITE);
    assert_eq!(b.result_for(WHITE), -1.0);
    assert_eq!(b.result_for(BLACK), 1.0);
}

#[test]
fn exact_five_rules_reject_an_overline() {
    let mut strict = Board::new(15, 5, false);
    let mut free = Board::new(15, 5, true);
    // Black fills the gap in `XXX.XX` to make six at once.  Going straight to
    // six matters: a run that passed through five would already have ended the
    // game under either rule, so it would not test the overline at all.
    // White's replies are spaced out so they never make a line of their own.
    for b in [&mut strict, &mut free] {
        for (i, &black) in [0usize, 1, 2, 4, 5].iter().enumerate() {
            assert!(b.play(7 * 15 + black), "black {black}");
            assert!(b.play(2 * i), "white filler {i}");
            assert!(!b.over, "no line of five appears while setting up");
        }
        b.play(7 * 15 + 3);
    }
    assert!(free.over, "six in a row wins freestyle");
    assert_eq!(free.winner, BLACK);
    assert_eq!(free.win_line().len(), 6);
    assert!(!strict.over, "six in a row is not an exact five");
    assert_eq!(strict.winner, EMPTY);
}

#[test]
fn undo_reverses_a_win() {
    let mut b = board();
    for i in 0..5 {
        b.play(7 * 15 + i);
        if i < 4 {
            b.play(i);
        }
    }
    assert!(b.over);
    assert!(b.undo());
    assert!(!b.over);
    assert_eq!(b.winner, EMPTY);
    assert_eq!(b.to_move, BLACK);
    assert_eq!(b.cells[7 * 15 + 4], EMPTY);
}

#[test]
fn undo_runs_back_to_the_empty_board() {
    let mut b = board();
    for m in [112usize, 113, 97, 128, 96] {
        b.play(m);
    }
    for _ in 0..5 {
        assert!(b.undo());
    }
    assert!(!b.undo(), "nothing left to undo");
    assert_eq!(b.to_move, BLACK);
    assert!(b.cells.iter().all(|&c| c == EMPTY));
}

#[test]
fn forced_move_prefers_winning_to_blocking() {
    let mut b = board();
    for i in 0..4usize {
        b.play(7 * 15 + i);
        b.play(9 * 15 + i);
    }
    assert_eq!(b.to_move, BLACK);
    assert_eq!(b.forced_move(), Some(7 * 15 + 4), "take the win, do not block");
}

#[test]
fn forced_move_blocks_when_there_is_no_win() {
    let mut b = board();
    for (i, black) in [0usize, 2, 4, 6].iter().enumerate() {
        b.play(*black);
        b.play(9 * 15 + i);
    }
    assert_eq!(b.to_move, BLACK);
    let forced = b.forced_move().expect("white's four must be blocked");
    assert!(forced == 9 * 15 + 4 || forced == 9 * 15 - 1, "got {forced}");
}

#[test]
fn forced_move_is_none_in_a_quiet_position() {
    let mut b = board();
    b.play(112);
    b.play(113);
    assert_eq!(b.forced_move(), None);
}

#[test]
fn the_neighbourhood_is_the_centre_on_an_empty_board() {
    let b = board();
    let near = b.neighbourhood(2);
    assert_eq!(near.iter().filter(|&&n| n).count(), 1);
    assert!(near[112]);
}

#[test]
fn the_neighbourhood_excludes_occupied_squares() {
    let mut b = board();
    b.play(112);
    let near = b.neighbourhood(1);
    assert!(!near[112], "the stone itself is not an empty neighbour");
    assert!(near[112 - 15] && near[112 + 15] && near[111] && near[113]);
    assert_eq!(near.iter().filter(|&&n| n).count(), 8);
}

#[test]
fn planes_are_written_from_the_side_to_moves_point_of_view() {
    let mut b = board();
    b.play(112); // black
    b.play(113); // white -- black to move again
    let area = 15 * 15;
    let mut planes = vec![0.0f32; NUM_PLANES * area];
    encode_board(&b, &mut planes);

    assert_eq!(planes[112], 1.0, "plane 0 holds the mover's own stones");
    assert_eq!(planes[area + 113], 1.0, "plane 1 holds the opponent's");
    assert_eq!(planes[2 * area + 113], 1.0, "plane 2 is the opponent's last move");
    assert_eq!(planes[3 * area + 112], 1.0, "plane 3 is the mover's previous move");
    assert!(planes[4 * area..].iter().all(|&v| v == 1.0), "plane 4 flags black to move");

    b.play(96); // now white is to move
    encode_board(&b, &mut planes);
    assert_eq!(planes[96], 0.0, "black's stones are now the opponent's");
    assert_eq!(planes[area + 96], 1.0);
    assert!(planes[4 * area..].iter().all(|&v| v == 0.0), "plane 4 is clear for white");
}

#[test]
fn a_batch_encodes_each_board_into_its_own_slice() {
    let mut a = board();
    a.play(112);
    let b = board();
    let planes = encode_batch(&[&a, &b]);
    let stride = NUM_PLANES * 225;
    assert_eq!(planes.len(), 2 * stride);
    assert_eq!(planes[stride + 112], 0.0, "the second board is empty");
    assert_eq!(planes[stride + 4 * 225], 1.0, "and has black to move");
}

#[test]
fn move_names_round_trip() {
    for index in [0usize, 14, 15, 112, 224] {
        let name = format_move(index, 15);
        assert_eq!(parse_move(&name, 15), Some(index), "{name}");
    }
    assert_eq!(parse_move("h7", 15), Some(112));
    assert_eq!(parse_move("7,7", 15), Some(112));
    assert_eq!(parse_move("112", 15), Some(112));
    assert_eq!(parse_move("z9", 15), None);
    assert_eq!(parse_move("", 15), None);
}

#[test]
fn a_filled_board_is_a_draw() {
    // 5x5 with a win length of 25 can never be won, so it fills up instead.
    let mut b = Board::new(5, 25, true);
    for i in 0..25 {
        assert!(b.play(i));
    }
    assert!(b.over);
    assert_eq!(b.winner, EMPTY);
    assert_eq!(b.result_for(BLACK), 0.0);
    assert!(b.legal_moves().is_empty());
}
