//! C ABI for the Omok AI core, consumed by the LuaJIT FFI bindings in `lua/`
//! and `love2d/`.
//!
//! Two things shape this interface.  First, MLX arrays belong to the thread
//! that made them, so the network is created on a worker thread and never
//! leaves it; the handle Lua holds is a pair of channels, not a network.
//! Second, LÖVE has to keep drawing at 60 fps while the engine thinks, so the
//! search is asynchronous by default: `begin` posts a position, `poll` returns
//! immediately with either "still thinking" or a result.
//!
//! Every fallible call reports failure by return value and leaves a message in
//! `omok_last_error()`, which is valid until the next failing call on the same
//! thread.

use std::ffi::{c_char, c_double, c_float, c_int, c_uchar, CStr, CString};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

use omok_core::board::Board;
use omok_core::net::Evaluator;
use omok_core::{search_interruptible, MctsConfig, Rng, SearchResult};
use omok_mlx::MlxNet;

// --------------------------------------------------------------- error state

thread_local! {
    static LAST_ERROR: std::cell::RefCell<CString> =
        std::cell::RefCell::new(CString::new("").unwrap());
}

fn set_error(message: impl Into<Vec<u8>>) {
    let text = CString::new(message).unwrap_or_else(|_| CString::new("error").unwrap());
    LAST_ERROR.with(|slot| *slot.borrow_mut() = text);
}

/// The message from the most recent failing call on this thread.
#[no_mangle]
pub extern "C" fn omok_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| slot.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn omok_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

// ---------------------------------------------------------------- game state

/// An Omok position with full move history.  Cheap, pure Rust, no engine
/// needed -- the TUI and the game both keep their board in one of these so the
/// rules are the same code the network was trained against.
pub struct OmokGame {
    board: Board,
}

fn game_ref<'a>(handle: *const OmokGame) -> Option<&'a Board> {
    unsafe { handle.as_ref().map(|g| &g.board) }
}

#[no_mangle]
pub extern "C" fn omok_game_new(size: c_int, win_length: c_int, allow_overline: c_int) -> *mut OmokGame {
    if size < 5 || size > 39 {
        set_error(format!("board size {size} is out of range (5..39)"));
        return std::ptr::null_mut();
    }
    if win_length < 3 || win_length > size {
        set_error(format!("win length {win_length} is out of range (3..{size})"));
        return std::ptr::null_mut();
    }
    Box::into_raw(Box::new(OmokGame {
        board: Board::new(size as usize, win_length as usize, allow_overline != 0),
    }))
}

#[no_mangle]
pub extern "C" fn omok_game_free(handle: *mut OmokGame) {
    if !handle.is_null() {
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// A deep copy, so the caller can explore a line without disturbing the game.
#[no_mangle]
pub extern "C" fn omok_game_clone(handle: *const OmokGame) -> *mut OmokGame {
    match game_ref(handle) {
        None => std::ptr::null_mut(),
        Some(board) => Box::into_raw(Box::new(OmokGame { board: board.clone() })),
    }
}

#[no_mangle]
pub extern "C" fn omok_game_reset(handle: *mut OmokGame) {
    if let Some(game) = unsafe { handle.as_mut() } {
        game.board.reset();
    }
}

/// 1 if the move was played, 0 if it was illegal or the game is over.
#[no_mangle]
pub extern "C" fn omok_game_play(handle: *mut OmokGame, index: c_int) -> c_int {
    match unsafe { handle.as_mut() } {
        None => 0,
        Some(game) => {
            if index < 0 {
                return 0;
            }
            game.board.play(index as usize) as c_int
        }
    }
}

#[no_mangle]
pub extern "C" fn omok_game_undo(handle: *mut OmokGame) -> c_int {
    match unsafe { handle.as_mut() } {
        None => 0,
        Some(game) => game.board.undo() as c_int,
    }
}

#[no_mangle]
pub extern "C" fn omok_game_is_legal(handle: *const OmokGame, index: c_int) -> c_int {
    match game_ref(handle) {
        None => 0,
        Some(board) => (index >= 0 && board.is_legal(index as usize)) as c_int,
    }
}

#[no_mangle]
pub extern "C" fn omok_game_size(handle: *const OmokGame) -> c_int {
    game_ref(handle).map(|b| b.size as c_int).unwrap_or(0)
}

/// 1 = black to move, 2 = white.
#[no_mangle]
pub extern "C" fn omok_game_to_move(handle: *const OmokGame) -> c_int {
    game_ref(handle).map(|b| b.to_move as c_int).unwrap_or(0)
}

/// 0 = nobody (draw or unfinished), 1 = black, 2 = white.
#[no_mangle]
pub extern "C" fn omok_game_winner(handle: *const OmokGame) -> c_int {
    game_ref(handle).map(|b| b.winner as c_int).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn omok_game_is_over(handle: *const OmokGame) -> c_int {
    game_ref(handle).map(|b| b.over as c_int).unwrap_or(0)
}

#[no_mangle]
pub extern "C" fn omok_game_move_count(handle: *const OmokGame) -> c_int {
    game_ref(handle).map(|b| b.moves.len() as c_int).unwrap_or(0)
}

/// The `i`-th move of the game, or -1.
#[no_mangle]
pub extern "C" fn omok_game_move_at(handle: *const OmokGame, i: c_int) -> c_int {
    match game_ref(handle) {
        None => -1,
        Some(board) => board
            .moves
            .get(i.max(0) as usize)
            .map(|&m| m as c_int)
            .unwrap_or(-1),
    }
}

#[no_mangle]
pub extern "C" fn omok_game_last_move(handle: *const OmokGame) -> c_int {
    game_ref(handle)
        .and_then(|b| b.last_move())
        .map(|m| m as c_int)
        .unwrap_or(-1)
}

/// Borrowed pointer to `size * size` bytes: 0 empty, 1 black, 2 white.  Valid
/// until the next call that changes the game.
#[no_mangle]
pub extern "C" fn omok_game_cells(handle: *const OmokGame) -> *const c_uchar {
    match game_ref(handle) {
        None => std::ptr::null(),
        Some(board) => board.cells.as_ptr(),
    }
}

/// Write the winning line into `out`, returning how many stones it holds.
#[no_mangle]
pub extern "C" fn omok_game_win_line(handle: *const OmokGame, out: *mut c_int, cap: c_int) -> c_int {
    let Some(board) = game_ref(handle) else { return 0 };
    if out.is_null() || cap <= 0 {
        return 0;
    }
    let line = board.win_line();
    let n = line.len().min(cap as usize);
    let slice = unsafe { std::slice::from_raw_parts_mut(out, n) };
    for (i, &index) in line.iter().take(n).enumerate() {
        slice[i] = index as c_int;
    }
    n as c_int
}

/// An immediate win for the side to move, else a square that must be blocked,
/// else -1.  The UI uses this for its hint button; the search uses it too.
#[no_mangle]
pub extern "C" fn omok_game_forced_move(handle: *const OmokGame) -> c_int {
    game_ref(handle)
        .and_then(|b| b.forced_move())
        .map(|m| m as c_int)
        .unwrap_or(-1)
}

// ------------------------------------------------------------------ searching

/// Mirrors `MctsConfig` so Lua can set difficulty without a dozen setters.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OmokOptions {
    pub simulations: c_int,
    pub c_puct: c_float,
    pub fpu_reduction: c_float,
    pub prior_local_radius: c_int,
    /// 0 plays the most-visited move; higher values sample for variety.
    pub temperature: c_float,
    /// Leaves gathered per network call.  0 keeps the default.
    pub batch: c_int,
}

impl From<OmokOptions> for MctsConfig {
    fn from(o: OmokOptions) -> MctsConfig {
        let d = MctsConfig::default();
        MctsConfig {
            simulations: if o.simulations > 0 { o.simulations as usize } else { d.simulations },
            c_puct: if o.c_puct > 0.0 { o.c_puct } else { d.c_puct },
            fpu_reduction: if o.fpu_reduction >= 0.0 { o.fpu_reduction } else { d.fpu_reduction },
            prior_local_radius: if o.prior_local_radius >= 0 {
                o.prior_local_radius as usize
            } else {
                d.prior_local_radius
            },
            temperature: o.temperature.max(0.0),
            batch: if o.batch > 0 { o.batch as usize } else { d.batch },
        }
    }
}

/// Fill in the defaults the shipped checkpoint was trained with.
#[no_mangle]
pub extern "C" fn omok_options_default(out: *mut OmokOptions) {
    let Some(out) = (unsafe { out.as_mut() }) else { return };
    let d = MctsConfig::default();
    *out = OmokOptions {
        simulations: d.simulations as c_int,
        c_puct: d.c_puct,
        fpu_reduction: d.fpu_reduction,
        prior_local_radius: d.prior_local_radius as c_int,
        temperature: d.temperature,
        batch: d.batch as c_int,
    };
}

pub const OMOK_TOP_MOVES: usize = 8;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OmokResult {
    pub best_move: c_int,
    /// Root value from the side-to-move's point of view, in [-1, 1].
    pub value: c_float,
    pub simulations: c_int,
    pub seconds: c_double,
    /// 1 when a one-ply win or block overrode the search's choice.
    pub forced: c_int,
    pub top_count: c_int,
    pub top_moves: [c_int; OMOK_TOP_MOVES],
    pub top_probs: [c_float; OMOK_TOP_MOVES],
}

impl OmokResult {
    fn from_search(r: &SearchResult) -> Self {
        let mut out = OmokResult {
            best_move: r.move_index as c_int,
            value: r.value,
            simulations: r.simulations as c_int,
            seconds: r.seconds,
            forced: r.forced as c_int,
            top_count: r.top.len().min(OMOK_TOP_MOVES) as c_int,
            top_moves: [-1; OMOK_TOP_MOVES],
            top_probs: [0.0; OMOK_TOP_MOVES],
        };
        for (i, &(m, p)) in r.top.iter().take(OMOK_TOP_MOVES).enumerate() {
            out.top_moves[i] = m as c_int;
            out.top_probs[i] = p;
        }
        out
    }
}

enum Command {
    Search { job: u64, board: Board, cfg: MctsConfig },
    Quit,
}

enum Event {
    Ready { description: String, board_size: usize },
    Failed(String),
    Done { job: u64, result: Box<SearchResult> },
    Error { job: u64, message: String },
}

/// The Lua-visible engine handle: channels to a worker thread that owns the
/// network, plus the job counter both sides use to agree on what is current.
pub struct OmokEngine {
    commands: Sender<Command>,
    events: Receiver<Event>,
    current: Arc<AtomicU64>,
    running: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    description: CString,
    board_size: usize,
    /// The most recent finished search, waiting to be collected by `poll`.
    ready: Option<OmokResult>,
    pending: bool,
}

fn worker_loop(
    path: PathBuf,
    seed: u64,
    commands: Receiver<Command>,
    events: Sender<Event>,
    current: Arc<AtomicU64>,
    running: Arc<AtomicBool>,
) {
    let mut net = match MlxNet::load(&path) {
        Ok(net) => net,
        Err(message) => {
            let _ = events.send(Event::Failed(message));
            return;
        }
    };
    let _ = events.send(Event::Ready {
        description: net.describe(),
        board_size: net.spec().board_size,
    });

    let mut rng = if seed == 0 { Rng::from_entropy() } else { Rng::new(seed) };
    while let Ok(command) = commands.recv() {
        let Command::Search { job, board, cfg } = command else { return };
        // A newer request landed while this one waited in the queue.
        if current.load(Ordering::SeqCst) != job {
            continue;
        }
        running.store(true, Ordering::SeqCst);
        let mut stop = || current.load(Ordering::SeqCst) != job;
        let outcome = search_interruptible(&board, &mut net, cfg, &mut rng, &mut stop);
        running.store(false, Ordering::SeqCst);
        if current.load(Ordering::SeqCst) != job {
            continue; // cancelled: drop the answer rather than deliver it late
        }
        let _ = match outcome {
            Ok(result) => events.send(Event::Done { job, result: Box::new(result) }),
            Err(message) => events.send(Event::Error { job, message }),
        };
    }
}

/// Load a `.npz` checkpoint and start the engine.  Blocks until the network is
/// loaded so that a bad path is an error here rather than a mystery later;
/// returns NULL and sets `omok_last_error()` on failure.
///
/// `seed` of 0 seeds the move-sampling RNG from the clock.
#[no_mangle]
pub extern "C" fn omok_engine_new(model_path: *const c_char, seed: u64) -> *mut OmokEngine {
    if model_path.is_null() {
        set_error("omok_engine_new: model path is NULL");
        return std::ptr::null_mut();
    }
    let path = match unsafe { CStr::from_ptr(model_path) }.to_str() {
        Ok(text) => PathBuf::from(text),
        Err(_) => {
            set_error("omok_engine_new: model path is not valid UTF-8");
            return std::ptr::null_mut();
        }
    };
    if !path.exists() {
        set_error(format!(
            "no model at {} -- train one with `make train`, or point OMOK_MODEL at a .npz checkpoint",
            path.display()
        ));
        return std::ptr::null_mut();
    }

    let (command_tx, command_rx) = channel();
    let (event_tx, event_rx) = channel();
    let current = Arc::new(AtomicU64::new(0));
    let running = Arc::new(AtomicBool::new(false));
    let worker = {
        let (current, running) = (current.clone(), running.clone());
        std::thread::Builder::new()
            .name("omok-engine".into())
            .spawn(move || worker_loop(path, seed, command_rx, event_tx, current, running))
    };
    let worker = match worker {
        Ok(handle) => handle,
        Err(e) => {
            set_error(format!("could not start the engine thread: {e}"));
            return std::ptr::null_mut();
        }
    };

    match event_rx.recv() {
        Ok(Event::Ready { description, board_size }) => Box::into_raw(Box::new(OmokEngine {
            commands: command_tx,
            events: event_rx,
            current,
            running,
            worker: Some(worker),
            description: CString::new(description).unwrap_or_default(),
            board_size,
            ready: None,
            pending: false,
        })),
        Ok(Event::Failed(message)) => {
            set_error(message);
            let _ = worker.join();
            std::ptr::null_mut()
        }
        _ => {
            set_error("the engine thread stopped before it was ready");
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn omok_engine_free(handle: *mut OmokEngine) {
    if handle.is_null() {
        return;
    }
    let mut engine = unsafe { Box::from_raw(handle) };
    engine.current.fetch_add(1, Ordering::SeqCst);
    let _ = engine.commands.send(Command::Quit);
    if let Some(worker) = engine.worker.take() {
        let _ = worker.join();
    }
}

/// Backend, architecture and checkpoint path, for the UI's status line.
#[no_mangle]
pub extern "C" fn omok_engine_describe(handle: *const OmokEngine) -> *const c_char {
    match unsafe { handle.as_ref() } {
        None => c"".as_ptr(),
        Some(engine) => engine.description.as_ptr(),
    }
}

/// The board size the network was trained for.
#[no_mangle]
pub extern "C" fn omok_engine_board_size(handle: *const OmokEngine) -> c_int {
    unsafe { handle.as_ref() }.map(|e| e.board_size as c_int).unwrap_or(0)
}

/// Start thinking about `game`.  Any search already running is abandoned.
/// Returns 1 on success, 0 if the position is finished or the call is invalid.
#[no_mangle]
pub extern "C" fn omok_search_begin(
    handle: *mut OmokEngine,
    game: *const OmokGame,
    options: *const OmokOptions,
) -> c_int {
    let Some(engine) = (unsafe { handle.as_mut() }) else {
        set_error("omok_search_begin: engine is NULL");
        return 0;
    };
    let Some(board) = game_ref(game) else {
        set_error("omok_search_begin: game is NULL");
        return 0;
    };
    if board.over {
        set_error("omok_search_begin: the game is already finished");
        return 0;
    }
    if board.size != engine.board_size {
        set_error(format!(
            "the board is {0}x{0} but the model was trained for {1}x{1}",
            board.size, engine.board_size
        ));
        return 0;
    }
    let cfg: MctsConfig = match unsafe { options.as_ref() } {
        Some(&o) => o.into(),
        None => MctsConfig::default(),
    };

    let job = engine.current.fetch_add(1, Ordering::SeqCst) + 1;
    engine.ready = None;
    engine.pending = true;
    if engine
        .commands
        .send(Command::Search { job, board: board.clone(), cfg })
        .is_err()
    {
        engine.pending = false;
        set_error("the engine thread has stopped");
        return 0;
    }
    1
}

/// Abandon whatever the engine is thinking about.  Its answer is dropped.
#[no_mangle]
pub extern "C" fn omok_search_cancel(handle: *mut OmokEngine) {
    if let Some(engine) = unsafe { handle.as_mut() } {
        engine.current.fetch_add(1, Ordering::SeqCst);
        engine.pending = false;
        engine.ready = None;
    }
}

/// 1 while the engine is thinking about a request that has not been collected.
#[no_mangle]
pub extern "C" fn omok_search_is_running(handle: *const OmokEngine) -> c_int {
    match unsafe { handle.as_ref() } {
        None => 0,
        Some(engine) => (engine.pending || engine.running.load(Ordering::SeqCst)) as c_int,
    }
}

fn drain(engine: &mut OmokEngine) -> Result<(), String> {
    let current = engine.current.load(Ordering::SeqCst);
    while let Ok(event) = engine.events.try_recv() {
        match event {
            Event::Done { job, result } if job == current => {
                engine.ready = Some(OmokResult::from_search(&result));
                engine.pending = false;
            }
            Event::Error { job, message } if job == current => {
                engine.pending = false;
                return Err(message);
            }
            _ => {} // a stale job's answer, or a duplicate ready notice
        }
    }
    Ok(())
}

/// Collect a finished search without blocking.
///
/// Returns 1 and fills `out` when a result is ready, 0 while the engine is
/// still thinking, and -1 on error (see `omok_last_error()`).
#[no_mangle]
pub extern "C" fn omok_search_poll(handle: *mut OmokEngine, out: *mut OmokResult) -> c_int {
    let Some(engine) = (unsafe { handle.as_mut() }) else {
        set_error("omok_search_poll: engine is NULL");
        return -1;
    };
    if let Err(message) = drain(engine) {
        set_error(message);
        return -1;
    }
    match engine.ready.take() {
        Some(result) => {
            if let Some(out) = unsafe { out.as_mut() } {
                *out = result;
            }
            1
        }
        None => 0,
    }
}

/// Think about `game` and wait for the answer.  Convenient for the TUI, which
/// has nothing to draw while it waits.  Returns 1 on success, 0 on error.
#[no_mangle]
pub extern "C" fn omok_search(
    handle: *mut OmokEngine,
    game: *const OmokGame,
    options: *const OmokOptions,
    out: *mut OmokResult,
) -> c_int {
    if omok_search_begin(handle, game, options) == 0 {
        return 0;
    }
    let Some(engine) = (unsafe { handle.as_mut() }) else { return 0 };
    let current = engine.current.load(Ordering::SeqCst);
    loop {
        match engine.events.recv() {
            Ok(Event::Done { job, result }) if job == current => {
                engine.pending = false;
                if let Some(out) = unsafe { out.as_mut() } {
                    *out = OmokResult::from_search(&result);
                }
                return 1;
            }
            Ok(Event::Error { job, message }) if job == current => {
                engine.pending = false;
                set_error(message);
                return 0;
            }
            Ok(_) => continue,
            Err(_) => {
                engine.pending = false;
                set_error("the engine thread has stopped");
                return 0;
            }
        }
    }
}
