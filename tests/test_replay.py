import numpy as np

from omok.board import BLACK, WHITE
from omok.replay import (GameRecord, ReplayBuffer, ShardWriter, count_games,
                         dataset_stats, shard_paths, write_shard)


def make_game(moves, winner, action_size=81, iteration=0, trainable=None):
    game = GameRecord(iteration=iteration, winner=winner)
    for i, move in enumerate(moves):
        policy = np.zeros(action_size, dtype=np.float32)
        policy[move] = 1.0
        use = True if trainable is None else trainable[i]
        game.add(move, policy, trainable=use)
    return game


def test_shard_roundtrip_and_counting(tmp_path):
    directory = str(tmp_path)
    write_shard(f"{directory}/shard-00000000.npz", [make_game([0, 1, 2], BLACK)])
    write_shard(f"{directory}/shard-00000001.npz",
                [make_game([3, 4], WHITE, iteration=1)])
    assert len(shard_paths(directory)) == 2
    assert count_games(directory) == 2
    assert count_games(directory, iteration=1) == 1
    stats = dataset_stats(directory)
    assert stats["games"] == 2 and stats["positions"] == 5


def test_writer_flushes_on_cadence(tmp_path):
    writer = ShardWriter(str(tmp_path), flush_every_games=2, flush_every_seconds=1e9)
    assert writer.add(make_game([0], BLACK)) is False   # buffered
    assert writer.add(make_game([1], WHITE)) is True    # flushed
    assert count_games(str(tmp_path)) == 2
    writer.add(make_game([2], BLACK))
    writer.close()                                      # flush on close
    assert count_games(str(tmp_path)) == 3


def test_writer_continues_numbering_after_restart(tmp_path):
    ShardWriter(str(tmp_path), flush_every_games=1).add(make_game([0], BLACK))
    ShardWriter(str(tmp_path), flush_every_games=1).add(make_game([1], WHITE))
    names = [p.split("/")[-1] for p in shard_paths(str(tmp_path))]
    assert names == ["shard-00000000.npz", "shard-00000001.npz"]


def test_value_targets_follow_the_side_to_move(tmp_path):
    # black plays moves 0, 2, 4...; black wins, so even plies are +1
    write_shard(f"{tmp_path}/shard-00000000.npz", [make_game([0, 1, 2, 3, 4], BLACK)])
    buffer = ReplayBuffer(9, 5, True, 1000).load_dir(str(tmp_path))
    data = buffer.merged()
    assert list(data["z"]) == [1.0, -1.0, 1.0, -1.0, 1.0]
    assert list(data["to_move"]) == [BLACK, WHITE, BLACK, WHITE, BLACK]


def test_positions_reconstruct_the_game(tmp_path):
    write_shard(f"{tmp_path}/shard-00000000.npz", [make_game([0, 1, 2], BLACK)])
    buffer = ReplayBuffer(9, 5, True, 1000).load_dir(str(tmp_path))
    data = buffer.merged()
    assert data["cells"][0].sum() == 0                 # empty board
    assert data["cells"][1][0] == BLACK                # after black's first move
    assert data["cells"][2][1] == WHITE
    assert data["last1"][2] == 1 and data["last2"][2] == 0


def test_sampling_shapes_and_augmentation(tmp_path):
    write_shard(f"{tmp_path}/shard-00000000.npz",
                [make_game(list(range(20)), BLACK) for _ in range(4)])
    buffer = ReplayBuffer(9, 5, True, 10_000).load_dir(str(tmp_path))
    rng = np.random.default_rng(0)
    planes, pi, z = buffer.sample(16, rng, augment=True)
    assert planes.shape == (16, 5, 9, 9)
    assert pi.shape == (16, 81) and z.shape == (16,)
    assert np.allclose(pi.sum(axis=1), 1.0, atol=1e-3)
    assert set(np.unique(planes[:, :4]).tolist()) <= {0.0, 1.0}
    for sample in planes:  # the colour plane stays constant under symmetries
        assert sample[4].min() == sample[4].max()


def test_untrainable_positions_are_never_sampled(tmp_path):
    flags = [False, False, True, True]
    write_shard(f"{tmp_path}/shard-00000000.npz",
                [make_game([0, 1, 2, 3], BLACK, trainable=flags)])
    buffer = ReplayBuffer(9, 5, True, 1000).load_dir(str(tmp_path))
    assert buffer.size == 4 and buffer.trainable_size == 2
    rng = np.random.default_rng(0)
    planes, _, _ = buffer.sample(64, rng, augment=False)
    # trainable positions have at least two stones on the board
    assert (planes[:, :2].sum(axis=(1, 2, 3)) >= 2).all()


def test_buffer_evicts_oldest_shards(tmp_path):
    for i in range(4):
        write_shard(f"{tmp_path}/shard-0000000{i}.npz", [make_game([0, 1, 2, 3, 4], BLACK)])
    buffer = ReplayBuffer(9, 5, True, max_positions=10).load_dir(str(tmp_path))
    assert buffer.size <= 15  # keeps only the newest shards that fit
    assert len(buffer._loaded) <= 3
    assert buffer._loaded[-1].endswith("shard-00000003.npz")


def test_corrupt_shard_is_skipped(tmp_path):
    write_shard(f"{tmp_path}/shard-00000000.npz", [make_game([0, 1], BLACK)])
    (tmp_path / "shard-00000001.npz").write_bytes(b"not an npz file")
    assert count_games(str(tmp_path)) == 1
    buffer = ReplayBuffer(9, 5, True, 1000).load_dir(str(tmp_path))
    assert buffer.size == 2
