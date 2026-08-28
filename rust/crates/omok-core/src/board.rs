//! Omok / gomoku rules -- a port of `omok/board.py`.
//!
//! The Python trainer is the source of truth for the rules, so this file
//! mirrors it move for move: same win test, same `forced_move` tactics, same
//! `to_move` flip on a terminal move (the search relies on `result_for` being
//! read from the point of view of the player who *would* have moved next).

pub const EMPTY: u8 = 0;
pub const BLACK: u8 = 1;
pub const WHITE: u8 = 2;
pub const DRAW: u8 = 0;

/// The four axes a line can run along.
const DIRECTIONS: [(i32, i32); 4] = [(0, 1), (1, 0), (1, 1), (1, -1)];

pub fn other(player: u8) -> u8 {
    if player == BLACK {
        WHITE
    } else {
        BLACK
    }
}

#[derive(Clone, Debug)]
pub struct Board {
    pub size: usize,
    pub win_length: usize,
    pub allow_overline: bool,
    pub cells: Vec<u8>,
    pub to_move: u8,
    pub moves: Vec<u16>,
    pub winner: u8,
    pub over: bool,
}

impl Board {
    pub fn new(size: usize, win_length: usize, allow_overline: bool) -> Self {
        Board {
            size,
            win_length,
            allow_overline,
            cells: vec![EMPTY; size * size],
            to_move: BLACK,
            moves: Vec::new(),
            winner: DRAW,
            over: false,
        }
    }

    pub fn action_size(&self) -> usize {
        self.size * self.size
    }

    pub fn move_number(&self) -> usize {
        self.moves.len()
    }

    pub fn last_move(&self) -> Option<usize> {
        self.moves.last().map(|&m| m as usize)
    }

    pub fn reset(&mut self) {
        self.cells.iter_mut().for_each(|c| *c = EMPTY);
        self.to_move = BLACK;
        self.moves.clear();
        self.winner = DRAW;
        self.over = false;
    }

    pub fn is_legal(&self, index: usize) -> bool {
        !self.over && index < self.cells.len() && self.cells[index] == EMPTY
    }

    pub fn legal_moves(&self) -> Vec<usize> {
        if self.over {
            return Vec::new();
        }
        (0..self.cells.len()).filter(|&i| self.cells[i] == EMPTY).collect()
    }

    pub fn legal_mask(&self) -> Vec<bool> {
        if self.over {
            return vec![false; self.cells.len()];
        }
        self.cells.iter().map(|&c| c == EMPTY).collect()
    }

    pub fn play(&mut self, index: usize) -> bool {
        if !self.is_legal(index) {
            return false;
        }
        let player = self.to_move;
        self.cells[index] = player;
        self.moves.push(index as u16);
        if self.wins_at(index, player) {
            self.winner = player;
            self.over = true;
        } else if self.moves.len() == self.cells.len() {
            self.winner = DRAW;
            self.over = true;
        }
        // The side to move always flips, terminal or not.
        self.to_move = other(player);
        true
    }

    /// Take back the last move.  Only that move could have ended the game, so
    /// undoing it always returns the position to "in progress".
    pub fn undo(&mut self) -> bool {
        match self.moves.pop() {
            None => false,
            Some(index) => {
                self.cells[index as usize] = EMPTY;
                self.to_move = other(self.to_move);
                self.winner = DRAW;
                self.over = false;
                true
            }
        }
    }

    /// Does a stone of `player` at `index` complete a winning line?  Never
    /// reads `index` itself, so it doubles as a "would this square win?" test.
    pub fn wins_at(&self, index: usize, player: u8) -> bool {
        let size = self.size as i32;
        let need = self.win_length;
        let row = (index / self.size) as i32;
        let col = (index % self.size) as i32;
        for (dr, dc) in DIRECTIONS {
            let mut count = 1usize;
            let (mut r, mut c) = (row + dr, col + dc);
            while r >= 0 && r < size && c >= 0 && c < size
                && self.cells[(r * size + c) as usize] == player
            {
                count += 1;
                r += dr;
                c += dc;
            }
            let (mut r, mut c) = (row - dr, col - dc);
            while r >= 0 && r < size && c >= 0 && c < size
                && self.cells[(r * size + c) as usize] == player
            {
                count += 1;
                r -= dr;
                c -= dc;
            }
            if count == need || (count > need && self.allow_overline) {
                return true;
            }
        }
        false
    }

    /// The stones making up the winning line, for the UI to highlight.
    pub fn win_line(&self) -> Vec<usize> {
        if !self.over || self.winner == DRAW {
            return Vec::new();
        }
        let Some(index) = self.last_move() else {
            return Vec::new();
        };
        let player = self.winner;
        let size = self.size as i32;
        let row = (index / self.size) as i32;
        let col = (index % self.size) as i32;
        for (dr, dc) in DIRECTIONS {
            let mut line = vec![index];
            let (mut r, mut c) = (row + dr, col + dc);
            while r >= 0 && r < size && c >= 0 && c < size
                && self.cells[(r * size + c) as usize] == player
            {
                line.push((r * size + c) as usize);
                r += dr;
                c += dc;
            }
            let (mut r, mut c) = (row - dr, col - dc);
            while r >= 0 && r < size && c >= 0 && c < size
                && self.cells[(r * size + c) as usize] == player
            {
                line.insert(0, (r * size + c) as usize);
                r -= dr;
                c -= dc;
            }
            if line.len() == self.win_length
                || (line.len() > self.win_length && self.allow_overline)
            {
                return line;
            }
        }
        Vec::new()
    }

    /// Empty squares where `player` would complete a five right now.
    pub fn winning_squares(&self, player: u8, candidates: &[usize]) -> Vec<usize> {
        if self.over {
            return Vec::new();
        }
        candidates
            .iter()
            .copied()
            .filter(|&i| self.cells[i] == EMPTY && self.wins_at(i, player))
            .collect()
    }

    /// One-ply tactics: take an immediate win, else block the opponent's.
    ///
    /// Search at casual simulation budgets can miss both, and missing either
    /// is what makes an engine look broken to a human.
    pub fn forced_move(&self) -> Option<usize> {
        if self.over {
            return None;
        }
        // A five-completing square always touches the stone next to the gap,
        // so the radius-2 neighbourhood of the stones is an exhaustive scan.
        let near = self.neighbourhood(2);
        let candidates: Vec<usize> =
            (0..near.len()).filter(|&i| near[i]).collect();
        let mine = self.winning_squares(self.to_move, &candidates);
        if let Some(&m) = mine.first() {
            return Some(m);
        }
        let theirs = self.winning_squares(other(self.to_move), &candidates);
        // Several at once is lost anyway; block one.
        theirs.first().copied()
    }

    pub fn result_for(&self, player: u8) -> f32 {
        if !self.over || self.winner == DRAW {
            0.0
        } else if self.winner == player {
            1.0
        } else {
            -1.0
        }
    }

    /// Mask of empty points within `radius` of an existing stone; the centre
    /// point on an empty board.
    pub fn neighbourhood(&self, radius: usize) -> Vec<bool> {
        let size = self.size;
        let mut near = vec![false; size * size];
        let mut any = false;
        for r in 0..size {
            for c in 0..size {
                if self.cells[r * size + c] == EMPTY {
                    continue;
                }
                any = true;
                let r0 = r.saturating_sub(radius);
                let r1 = (r + radius + 1).min(size);
                let c0 = c.saturating_sub(radius);
                let c1 = (c + radius + 1).min(size);
                for rr in r0..r1 {
                    for cc in c0..c1 {
                        near[rr * size + cc] = true;
                    }
                }
            }
        }
        if !any {
            near[size / 2 * size + size / 2] = true;
            return near;
        }
        for i in 0..near.len() {
            near[i] = near[i] && self.cells[i] == EMPTY;
        }
        near
    }

    pub fn to_ascii(&self) -> String {
        let size = self.size;
        let mut out = String::from("    ");
        for c in 0..size {
            out.push_str(&format!("{} ", c % 10));
        }
        for r in 0..size {
            out.push_str(&format!("\n{:>3} ", r));
            for c in 0..size {
                out.push(match self.cells[r * size + c] {
                    BLACK => 'X',
                    WHITE => 'O',
                    _ => '.',
                });
                out.push(' ');
            }
        }
        out
    }
}

/// `h8` / `H8`, `7,7`, or a raw index -- the same grammar as `parse_move` in
/// `omok/board.py`.
pub fn parse_move(text: &str, size: usize) -> Option<usize> {
    let text: String = text.trim().to_lowercase().replace(' ', "");
    if text.is_empty() {
        return None;
    }
    if let Some((row, col)) = text.split_once(',') {
        let r: usize = row.parse().ok()?;
        let c: usize = col.parse().ok()?;
        return (r < size && c < size).then_some(r * size + c);
    }
    let first = text.chars().next()?;
    if first.is_ascii_alphabetic() {
        let col = (first as u8 - b'a') as usize;
        let row: usize = text[1..].parse().ok()?;
        return (row < size && col < size).then_some(row * size + col);
    }
    let value: usize = text.parse().ok()?;
    (value < size * size).then_some(value)
}

pub fn format_move(index: usize, size: usize) -> String {
    let (row, col) = (index / size, index % size);
    format!("{}{}", (b'a' + col as u8) as char, row)
}
