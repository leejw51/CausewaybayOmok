//! Load a checkpoint, print what it thinks of the empty board, and time a
//! search.  Handy for checking parity against the Python engine.
//!
//!     cargo run --release --example bench -- runs/blitz/checkpoints/best.npz

use omok_core::board::Board;
use omok_core::net::Evaluator;
use omok_core::{format_move, search, MctsConfig, Rng};
use omok_mlx::MlxNet;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "runs/blitz/checkpoints/best.npz".to_string());
    let mut net = match MlxNet::load(std::path::Path::new(&path)) {
        Ok(net) => net,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    println!("{}", net.describe());

    let spec = net.spec();
    let mut board = Board::new(spec.board_size, 5, true);
    let planes = omok_core::encode::encode_batch(&[&board]);
    let prediction = net.predict(&planes, 1).unwrap();
    let best = prediction
        .policy_row(0)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();
    println!(
        "empty board: value {:+.4}, best prior {} p={:.4}",
        prediction.value[0],
        format_move(best.0, spec.board_size),
        best.1
    );

    let mut rng = Rng::new(1234);
    let cfg = MctsConfig::default();
    for ply in 0..6 {
        let result = search(&board, &mut net, cfg, &mut rng).unwrap();
        println!(
            "ply {ply}: {} value {:+.3} {} sims in {:.3}s ({:.0} sims/s){}",
            format_move(result.move_index, spec.board_size),
            result.value,
            result.simulations,
            result.seconds,
            result.simulations as f64 / result.seconds.max(1e-9),
            if result.forced { " [forced]" } else { "" }
        );
        board.play(result.move_index);
    }
    println!("{}", board.to_ascii());
}
