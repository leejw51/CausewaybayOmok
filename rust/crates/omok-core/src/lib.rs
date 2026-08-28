//! The engine-independent half of the Omok AI core: rules, plane encoding,
//! checkpoint loading and search.  The network itself lives in `omok-mlx`.

pub mod board;
pub mod encode;
pub mod mcts;
pub mod net;
pub mod netspec;
pub mod npz;
pub mod rng;

pub use board::{format_move, parse_move, Board, BLACK, DRAW, EMPTY, WHITE};
pub use mcts::{search, search_interruptible, MctsConfig, SearchResult};
pub use net::{Evaluator, Prediction};
pub use netspec::NetSpec;
pub use rng::Rng;
