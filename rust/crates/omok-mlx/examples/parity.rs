//! Print raw predictions for a fixed set of positions, so they can be compared
//! against the Python backend's numbers for the same checkpoint.

use omok_core::board::Board;
use omok_core::net::Evaluator;
use omok_mlx::MlxNet;

fn main() {
    let path = std::env::args().nth(1).unwrap();
    let mut net = MlxNet::load(std::path::Path::new(&path)).unwrap();
    let mut board = Board::new(15, 5, true);
    let mut boards = vec![board.clone()];
    for m in [112usize, 113, 97, 128, 96] {
        board.play(m);
        boards.push(board.clone());
    }
    let refs: Vec<&Board> = boards.iter().collect();
    let planes = omok_core::encode::encode_batch(&refs);
    let p = net.predict(&planes, refs.len()).unwrap();
    for i in 0..refs.len() {
        let row = p.policy_row(i);
        let (arg, max) = row.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap();
        println!("pos {i}: value {:+.6} argmax {arg} p={max:.6}", p.value[i]);
    }
}
