import numpy as np
import pytest

from omok.effects import (Animated, Clip, Particles, clamp, ease_in_out_cubic,
                          ease_out_back, ease_out_cubic, ease_out_elastic,
                          ease_out_quad, linear, mix, pulse)

CURVES = (linear, ease_out_quad, ease_out_cubic, ease_in_out_cubic, ease_out_back,
          ease_out_elastic)


# ------------------------------------------------------------------ easing
@pytest.mark.parametrize("curve", CURVES)
def test_every_curve_starts_at_zero_and_ends_at_one(curve):
    assert curve(0.0) == pytest.approx(0.0, abs=1e-6)
    assert curve(1.0) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("curve", CURVES)
def test_curves_stay_in_a_sane_range(curve):
    """Overshoot is allowed -- that is the point of back/elastic -- but not wildly."""
    values = [curve(t / 40.0) for t in range(41)]
    assert min(values) > -0.35
    assert max(values) < 1.45


def test_ease_out_curves_front_load_their_motion():
    """An 'out' curve is more than half done at the halfway point."""
    for curve in (ease_out_quad, ease_out_cubic, ease_out_back):
        assert curve(0.5) > 0.5


def test_ease_out_back_overshoots_before_settling():
    values = [ease_out_back(t / 100.0) for t in range(101)]
    assert max(values) > 1.0          # it goes past the target
    assert values[-1] == pytest.approx(1.0, abs=1e-6)


def test_pulse_breathes_between_zero_and_one():
    assert pulse(0.0) == pytest.approx(0.0, abs=1e-6)
    assert pulse(0.7, period=1.4) == pytest.approx(1.0, abs=1e-6)
    assert all(0.0 <= pulse(t / 10.0) <= 1.0 for t in range(60))


def test_clamp_and_mix():
    assert clamp(-2.0) == 0.0 and clamp(2.0) == 1.0 and clamp(0.25) == 0.25
    assert clamp(5.0, 1.0, 3.0) == 3.0
    assert mix(10.0, 20.0, 0.5) == 15.0


# ------------------------------------------------------------------ tweens
def test_animated_eases_to_its_target():
    value = Animated(0.0, duration=1.0, ease=linear)
    value.set(10.0)
    value.update(0.5)
    assert value.value == pytest.approx(5.0)
    assert not value.done
    value.update(0.5)
    assert value.value == pytest.approx(10.0)
    assert value.done


def test_animated_retargets_from_where_it_currently_is():
    value = Animated(0.0, duration=1.0, ease=linear)
    value.set(10.0)
    value.update(0.5)          # half way, at 5.0
    value.set(0.0)             # turn around
    value.update(0.5)
    assert value.value == pytest.approx(2.5)


def test_animated_snap_skips_the_animation():
    value = Animated(1.0)
    value.set(9.0, snap=True)
    assert value.value == 9.0 and value.done


def test_clip_reports_progress_then_finishes():
    clip = Clip(0.5, data="x")
    assert clip.update(0.25) is True
    assert clip.progress == pytest.approx(0.5)
    assert clip.update(0.25) is False
    assert clip.done and clip.progress == 1.0
    assert clip.data == "x"


def test_zero_length_clip_is_immediately_done():
    assert Clip(0.0).done and Clip(0.0).progress == 1.0


# --------------------------------------------------------------- particles
def emit_some(particles, count=10, **kwargs):
    options = dict(colour=(255, 128, 0), speed=(10.0, 20.0), life=(1.0, 1.0),
                   size=(2.0, 3.0))
    options.update(kwargs)
    particles.emit(count, 100.0, 100.0, **options)


def test_particles_expire_and_are_compacted():
    particles = Particles(capacity=64, seed=0)
    emit_some(particles, 10, life=(0.5, 0.5))
    emit_some(particles, 10, life=(2.0, 2.0))
    assert len(particles) == 20
    particles.update(1.0)          # the first batch is past its life
    assert len(particles) == 10
    # The survivors are the long-lived ones, and their state moved with them.
    assert np.all(particles.life[:10] == 2.0)
    particles.update(2.0)
    assert len(particles) == 0


def test_particles_never_exceed_capacity():
    particles = Particles(capacity=16, seed=0)
    emit_some(particles, 100)
    assert len(particles) == 16
    emit_some(particles, 100)      # already full: emitting is a no-op
    assert len(particles) == 16


def test_particles_move_and_slow_down():
    particles = Particles(capacity=8, seed=1)
    emit_some(particles, 4, speed=(100.0, 100.0), drag=2.0)
    speed_before = np.hypot(particles.vx[:4], particles.vy[:4]).mean()
    particles.update(0.1)
    speed_after = np.hypot(particles.vx[:4], particles.vy[:4]).mean()
    assert speed_after < speed_before
    assert not np.allclose(particles.x[:4], 100.0)


def test_gravity_pulls_particles_down():
    particles = Particles(capacity=4, seed=2)
    emit_some(particles, 2, speed=(0.0, 0.0), gravity=500.0, drag=0.0)
    particles.update(0.1)
    assert np.all(particles.vy[:2] < 0.0)
    assert np.all(particles.y[:2] < 100.0)


def test_a_directed_burst_goes_where_it_is_aimed():
    particles = Particles(capacity=32, seed=3)
    emit_some(particles, 16, speed=(50.0, 50.0), direction=np.pi / 2, arc=0.4)
    assert np.all(particles.vy[:16] > 0.0)                 # upward
    assert np.all(np.abs(particles.vx[:16]) < 20.0)        # and narrow


def test_view_reports_fading_alpha_and_growing_radius():
    particles = Particles(capacity=8, seed=4)
    emit_some(particles, 4, life=(1.0, 1.0), size=(10.0, 10.0), grow=1.0)
    _, _, radius, alpha, _, _, _, _ = particles.view()
    assert np.allclose(alpha, 1.0) and np.allclose(radius, 10.0)
    particles.update(0.5)
    _, _, radius, alpha, _, _, _, _ = particles.view()
    assert np.allclose(alpha, 0.5, atol=1e-5)
    assert np.allclose(radius, 15.0, atol=1e-4)


def test_clear_empties_the_pool():
    particles = Particles(capacity=8, seed=5)
    emit_some(particles, 6)
    particles.clear()
    assert len(particles) == 0
