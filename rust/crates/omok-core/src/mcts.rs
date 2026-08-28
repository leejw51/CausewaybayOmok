//! Batched PUCT search -- the Rust side of `omok/mcts.py`.
//!
//! The trainer batches its network calls across *games*: many self-play games
//! search at once, each contributing one leaf per iteration.  A game against a
//! human has only one position to think about, so this port batches across
//! *leaves* instead: it descends `batch` times before calling the network,
//! holding a virtual loss on each path so the descents pick different leaves.
//! Same arithmetic, same priors, same one-ply safety net -- but MLX sees
//! batches of 16 rather than a stream of single positions, which is the
//! difference between a search that feels instant and one that does not.

use crate::board::{Board, EMPTY};
use crate::net::Evaluator;
use crate::rng::Rng;

#[derive(Clone, Copy, Debug)]
pub struct MctsConfig {
    pub simulations: usize,
    pub c_puct: f32,
    pub fpu_reduction: f32,
    /// Zero the prior of moves farther than this from every stone (0 = off).
    pub prior_local_radius: usize,
    /// 0 = always play the most-visited move; higher samples more freely.
    pub temperature: f32,
    /// Leaves collected per network call.
    pub batch: usize,
}

impl Default for MctsConfig {
    fn default() -> Self {
        // The values the shipped checkpoint was trained with (runs/blitz).
        MctsConfig {
            simulations: 160,
            c_puct: 1.6,
            fpu_reduction: 0.25,
            prior_local_radius: 2,
            temperature: 0.0,
            batch: 16,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SearchResult {
    pub move_index: usize,
    /// Root value from the side-to-move's point of view, in [-1, 1].
    pub value: f32,
    pub simulations: usize,
    pub seconds: f64,
    /// Up to eight (move, visit share) pairs, best first.
    pub top: Vec<(usize, f32)>,
    /// True when a one-ply win or block overrode the search's choice.
    pub forced: bool,
}

#[derive(Default)]
struct Node {
    p: Vec<f32>,
    n: Vec<i32>,
    w: Vec<f32>,
    vloss: Vec<i32>,
    /// 0.0 on legal moves, -inf on illegal ones, so scoring stays one add.
    bias: Vec<f32>,
    /// 1.0 while a child has never been visited, 0.0 after.
    unvisited: Vec<f32>,
    /// action -> arena index, or `usize::MAX` for "not created yet".
    children: Vec<usize>,
    expanded: bool,
    terminal: bool,
    visits: i32,
    vloss_total: i32,
    value: f32,
    w_sum: f64,
    p_explored: f64,
}

const NO_CHILD: usize = usize::MAX;

impl Node {
    fn expand(&mut self, priors: &[f32], legal: &[bool]) {
        let action_size = priors.len();
        let mut masked: Vec<f32> = (0..action_size)
            .map(|i| if legal[i] { priors[i].max(0.0) } else { 0.0 })
            .collect();
        let mut total: f32 = masked.iter().sum();
        if total <= 1e-8 {
            // The network gave all its mass to illegal moves: fall back to
            // uniform over what is actually playable.
            masked = legal.iter().map(|&l| l as u8 as f32).collect();
            total = masked.iter().sum();
        }
        let scale = 1.0 / total.max(1e-8);
        self.p = masked.into_iter().map(|v| v * scale).collect();
        self.n = vec![0; action_size];
        self.w = vec![0.0; action_size];
        self.vloss = vec![0; action_size];
        self.bias = legal
            .iter()
            .map(|&l| if l { 0.0 } else { f32::NEG_INFINITY })
            .collect();
        self.unvisited = vec![1.0; action_size];
        self.children = vec![NO_CHILD; action_size];
        self.w_sum = 0.0;
        self.p_explored = 0.0;
        self.expanded = true;
    }
}

pub struct Search<'a> {
    cfg: MctsConfig,
    nodes: Vec<Node>,
    board: Board,
    rng: &'a mut Rng,
}

impl<'a> Search<'a> {
    pub fn new(board: Board, cfg: MctsConfig, rng: &'a mut Rng) -> Self {
        Search { cfg, nodes: vec![Node::default()], board, rng }
    }

    fn root(&self) -> usize {
        0
    }

    /// PUCT: argmax over Q + U, unvisited children held at the first-play
    /// urgency, illegal moves at -inf.  Virtual losses count as visits that
    /// lost, which is what keeps a batch of descents from piling onto one leaf.
    fn select_action(&self, idx: usize) -> usize {
        let node = &self.nodes[idx];
        let visits = node.visits + node.vloss_total;
        let total = if visits > 0 { visits } else { 1 } as f32;
        let sqrt_total = total.sqrt();
        let parent_q = if node.visits > 0 {
            (node.w_sum / node.visits as f64) as f32
        } else {
            0.0
        };
        let fpu = parent_q - self.cfg.fpu_reduction * (node.p_explored.max(0.0) as f32).sqrt();

        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for a in 0..node.p.len() {
            if node.bias[a] == f32::NEG_INFINITY {
                continue;
            }
            let nf = (node.n[a] + node.vloss[a]) as f32;
            let w = node.w[a] - node.vloss[a] as f32;
            let q = w / nf.max(1.0);
            let u = self.cfg.c_puct * node.p[a] / (nf + 1.0) * sqrt_total;
            let mut score = q + u;
            if fpu != 0.0 {
                score += node.unvisited[a] * fpu;
            }
            if score > best_score {
                best_score = score;
                best = a;
            }
        }
        best
    }

    /// Walk to a leaf, creating nodes as needed.  Returns the leaf and the
    /// position it stands for, plus the (node, action) path taken.
    fn descend(&mut self) -> (usize, Board, Vec<(usize, usize)>) {
        let mut idx = self.root();
        let mut board = self.board.clone();
        let mut path = Vec::new();
        while self.nodes[idx].expanded && !self.nodes[idx].terminal {
            let action = self.select_action(idx);
            path.push((idx, action));
            board.play(action);
            let mut child = self.nodes[idx].children[action];
            if child == NO_CHILD {
                child = self.nodes.len();
                self.nodes.push(Node::default());
                self.nodes[idx].children[action] = child;
            }
            idx = child;
            if board.over {
                self.nodes[idx].terminal = true;
                self.nodes[idx].value = board.result_for(board.to_move);
                break;
            }
        }
        (idx, board, path)
    }

    fn add_virtual_loss(&mut self, path: &[(usize, usize)]) {
        for &(idx, action) in path {
            let node = &mut self.nodes[idx];
            node.vloss[action] += 1;
            node.vloss_total += 1;
        }
    }

    fn remove_virtual_loss(&mut self, path: &[(usize, usize)]) {
        for &(idx, action) in path {
            let node = &mut self.nodes[idx];
            node.vloss[action] -= 1;
            node.vloss_total -= 1;
        }
    }

    /// The leaf value is from the leaf mover's point of view, so it flips sign
    /// on every step back towards the root.
    fn backup(&mut self, path: &[(usize, usize)], value: f32) {
        let mut sign = -1.0f32;
        for &(idx, action) in path.iter().rev() {
            let node = &mut self.nodes[idx];
            if node.n[action] == 0 {
                node.unvisited[action] = 0.0;
                node.p_explored += node.p[action] as f64;
            }
            node.n[action] += 1;
            let delta = sign * value;
            node.w[action] += delta;
            node.w_sum += delta as f64;
            node.visits += 1;
            sign = -sign;
        }
    }

    /// Restrict the priors to the neighbourhood of the stones.  In gomoku
    /// every tactically relevant move is next to the action, so this points the
    /// whole simulation budget at the part of the board that matters.
    fn localise(&self, board: &Board, priors: &mut [f32]) {
        if self.cfg.prior_local_radius == 0 {
            return;
        }
        let near = board.neighbourhood(self.cfg.prior_local_radius);
        let kept: f32 = (0..priors.len()).map(|i| if near[i] { priors[i] } else { 0.0 }).sum();
        if kept > 1e-6 {
            for i in 0..priors.len() {
                if !near[i] {
                    priors[i] = 0.0;
                }
            }
        }
    }

    /// `stop` is polled once per network call, so a search can be abandoned
    /// the moment the player undoes a move or starts a new game.
    pub fn run(
        &mut self,
        evaluator: &mut dyn Evaluator,
        stop: &mut dyn FnMut() -> bool,
    ) -> Result<usize, String> {
        if self.board.over {
            return Ok(0);
        }
        let mut done = 0usize;
        let target = self.cfg.simulations.max(1);
        let batch = self.cfg.batch.max(1);

        while done < target {
            if stop() {
                break;
            }
            let want = batch.min(target - done);
            let mut pending: Vec<(usize, Board, Vec<(usize, usize)>)> = Vec::with_capacity(want);
            let mut claimed: Vec<usize> = Vec::with_capacity(want);
            let before = done;

            while pending.len() < want {
                let (leaf, board, path) = self.descend();
                if self.nodes[leaf].terminal {
                    let value = self.nodes[leaf].value;
                    self.backup(&path, value);
                    done += 1;
                    if done >= target {
                        break;
                    }
                    continue;
                }
                if claimed.contains(&leaf) {
                    // The virtual loss did not push this descent onto a
                    // different leaf, which happens when a node has only one
                    // move worth making.  Evaluate what has been gathered
                    // rather than expanding the same position twice.
                    break;
                }
                claimed.push(leaf);
                self.add_virtual_loss(&path);
                pending.push((leaf, board, path));
            }

            if pending.is_empty() {
                // Nothing to evaluate and nothing backed up: the tree cannot
                // grow any further from here, so stop instead of spinning.
                if done == before {
                    break;
                }
                continue;
            }

            let boards: Vec<&Board> = pending.iter().map(|(_, b, _)| b).collect();
            let planes = crate::encode::encode_batch(&boards);
            let prediction = evaluator.predict(&planes, boards.len())?;

            for (i, (leaf, board, path)) in pending.into_iter().enumerate() {
                self.remove_virtual_loss(&path);
                let mut priors = prediction.policy_row(i).to_vec();
                self.localise(&board, &mut priors);
                let value = prediction.value[i];
                let legal = board.legal_mask();
                self.nodes[leaf].expand(&priors, &legal);
                self.nodes[leaf].value = value;
                self.backup(&path, value);
                done += 1;
            }
        }
        Ok(done)
    }

    pub fn visit_distribution(&self) -> Vec<f32> {
        let root = &self.nodes[self.root()];
        if !root.expanded {
            let legal = self.board.legal_mask();
            let total = legal.iter().filter(|&&l| l).count().max(1) as f32;
            return legal.iter().map(|&l| l as u8 as f32 / total).collect();
        }
        let total: f32 = root.n.iter().map(|&n| n as f32).sum();
        if total <= 0.0 {
            let legal = self.board.legal_mask();
            let count = legal.iter().filter(|&&l| l).count().max(1) as f32;
            return legal.iter().map(|&l| l as u8 as f32 / count).collect();
        }
        root.n.iter().map(|&n| n as f32 / total).collect()
    }

    pub fn root_value(&self) -> f32 {
        let root = &self.nodes[self.root()];
        if !root.expanded || root.visits == 0 {
            return root.value;
        }
        (root.w_sum / root.visits.max(1) as f64) as f32
    }

    fn pick_move(&mut self, probs: &[f32]) -> usize {
        // The *first* maximum, as `np.argmax` returns and so as the Python
        // trainer picks.  It also keeps the chosen move in step with the top
        // list `finish` reports, which sorts stably: at a small simulation
        // budget a lot of moves tie on visits, and `max_by`'s last-maximum
        // would name a move that does not appear in the list at all.
        let argmax = |probs: &[f32]| {
            let mut best = 0usize;
            let mut best_prob = f32::NEG_INFINITY;
            for i in 0..probs.len() {
                if self.board.cells[i] == EMPTY && probs[i] > best_prob {
                    best_prob = probs[i];
                    best = i;
                }
            }
            best
        };
        if self.cfg.temperature <= 1e-3 {
            return argmax(probs);
        }
        let inv = 1.0 / self.cfg.temperature;
        let scaled: Vec<f32> = probs.iter().map(|&p| p.powf(inv)).collect();
        let total: f32 = scaled.iter().sum();
        if !(total > 0.0) {
            return argmax(probs);
        }
        let mut cursor = self.rng.next_f32() * total;
        for (i, &s) in scaled.iter().enumerate() {
            cursor -= s;
            if cursor <= 0.0 {
                return i;
            }
        }
        argmax(probs)
    }

    pub fn finish(mut self, simulations: usize, seconds: f64) -> SearchResult {
        let probs = self.visit_distribution();
        let mut order: Vec<usize> = (0..probs.len()).filter(|&i| probs[i] > 0.0).collect();
        order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<(usize, f32)> = order.into_iter().take(8).map(|i| (i, probs[i])).collect();

        let value = self.root_value();
        // Never miss a one-ply win or fail to block one, whatever the search
        // says -- this is the difference between "beatable" and "broken".
        let forced = self.board.forced_move();
        let move_index = match forced {
            Some(m) => m,
            None => self.pick_move(&probs),
        };
        SearchResult {
            move_index,
            value,
            simulations,
            seconds,
            top,
            forced: forced.is_some(),
        }
    }
}

/// Search `board` and return the move to play.
pub fn search(
    board: &Board,
    evaluator: &mut dyn Evaluator,
    cfg: MctsConfig,
    rng: &mut Rng,
) -> Result<SearchResult, String> {
    search_interruptible(board, evaluator, cfg, rng, &mut || false)
}

/// As [`search`], but abandons the search as soon as `stop` returns true.
/// Returns `None` if it was stopped before a single simulation finished.
pub fn search_interruptible(
    board: &Board,
    evaluator: &mut dyn Evaluator,
    cfg: MctsConfig,
    rng: &mut Rng,
    stop: &mut dyn FnMut() -> bool,
) -> Result<SearchResult, String> {
    let started = std::time::Instant::now();
    let mut search = Search::new(board.clone(), cfg, rng);
    let simulations = search.run(evaluator, stop)?;
    Ok(search.finish(simulations, started.elapsed().as_secs_f64()))
}
