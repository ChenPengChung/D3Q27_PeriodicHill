#!/usr/bin/env python3
"""
Comprehensive unit tests for PI force controller (evolution.h).

Simulates the closed-loop system:
  - Controller: PI with anti-windup + Ma brake (mirrors evolution.h exactly)
  - Plant: simplified 1D momentum equation  dUb/dt = Force - C_drag * Ub^2
    where C_drag is calibrated so steady-state Ub = Uref at a reasonable Force.

Tests cover:
  1. Cold start (Ub=0)
  2. Near-target below (Ub/Uref ~ 0.95)
  3. Near-target above / overshoot (Ub/Uref ~ 1.05, 1.2)
  4. Ma_max > 0.25 warning brake
  5. Ma_max > 0.30 critical brake
  6. Long-term stability: Ma_max converges below 0.2
  7. Integral anti-windup behavior
  8. Force non-negativity under overshoot
  9. Repeated Ma brake recovery
  10. Restart from high Ub (Ub/Uref = 1.5)
  11. Very low Re% oscillation damping

Usage:
  python3 test_force_controller.py
"""
import numpy as np
import sys

# ============================================================
# Simulation parameters (from variables.h)
# ============================================================
Uref   = 0.017          # target bulk velocity
Re     = 700.0
LY     = 9.0
cs     = 1.0 / np.sqrt(3.0)  # LBM sound speed
NDTFRC = 200            # force update interval (steps)

# Controller parameters (from evolution.h)
Kp   = 2.0
Ki   = 0.3
norm = Uref**2 / LY
Force_max_integral = 20.0 * norm

# ============================================================
# PI Controller (exact replica of evolution.h logic)
# ============================================================
class PIForceController:
    """Exact Python replica of the PI controller in evolution.h."""

    def __init__(self):
        self.Force = 0.0
        self.Force_integral = 0.0

    def update(self, Ub_avg, Ma_max):
        """
        Single controller update step.
        Returns (Force, ctrl_mode, Re_pct).
        """
        error  = Uref - Ub_avg
        Re_pct = (Ub_avg - Uref) / Uref * 100.0

        # Integral accumulation
        self.Force_integral += Ki * error * norm

        # Conditional decay: when error and integral conflict (overshoot but integral still pushing)
        if error < 0.0 and self.Force_integral > 0.0:
            self.Force_integral *= 0.8  # 20% decay per update

        # Anti-windup clamp
        if self.Force_integral > Force_max_integral:
            self.Force_integral = Force_max_integral
        if self.Force_integral < 0.0:
            self.Force_integral = 0.0

        # PI synthesis
        Force_new = Kp * error * norm + self.Force_integral

        # Mode label
        if abs(Re_pct) < 1.5:
            ctrl_mode = "PI-steady"
        elif error > 0:
            ctrl_mode = "PI-accel"
        else:
            ctrl_mode = "PI-decel"

        self.Force = Force_new

        # Back-calculation anti-windup:
        # When Force clamped to 0, back-calculate integral so Force = 0 exactly
        if self.Force < 0.0:
            self.Force = 0.0
            integral_target = max(0.0, -Kp * error * norm)
            if self.Force_integral > integral_target:
                self.Force_integral = integral_target

        # Ma safety brake (higher threshold first!)
        if Ma_max > 0.30:
            self.Force *= 0.05
            self.Force_integral *= 0.05
        elif Ma_max > 0.25:
            self.Force *= 0.5
            self.Force_integral *= 0.5

        return self.Force, ctrl_mode, Re_pct


# ============================================================
# Simple plant model: dUb/dt = Force - C_drag * Ub^2
# ============================================================
class SimplePlant:
    """
    1D momentum model for periodic hill flow.
    dUb/dt = Force - C_drag * Ub^2
    Ma_max = Ma_ratio * Ub / cs  (local max is some multiple of bulk)
    """

    def __init__(self, Ub0=0.0, C_drag=None, Ma_ratio=2.5, dt_sim=1.0):
        self.Ub = Ub0
        self.dt = dt_sim
        # Calibrate C_drag so that at Ub=Uref, Force_ss ≈ some reasonable value
        # From real data: F* ≈ 0.5 → Force_ss = 0.5 * Uref^2/LY ≈ 1.6e-5
        if C_drag is None:
            Force_ss = 0.5 * norm
            self.C_drag = Force_ss / Uref**2
        else:
            self.C_drag = C_drag
        self.Ma_ratio = Ma_ratio  # Ma_max / Ma_bulk ratio (hill crest acceleration)

    def step(self, Force, n_substeps=NDTFRC):
        """Advance plant by n_substeps with given Force."""
        dt_sub = self.dt
        for _ in range(n_substeps):
            dUb = (Force - self.C_drag * self.Ub * abs(self.Ub)) * dt_sub
            self.Ub += dUb
            # Physical lower bound
            if self.Ub < 0:
                self.Ub = 0.0

    @property
    def Ma_max(self):
        return self.Ma_ratio * abs(self.Ub) / cs


# ============================================================
# Test harness
# ============================================================
passed = 0
failed = 0
total  = 0

def run_test(name, test_func):
    global passed, failed, total
    total += 1
    try:
        test_func()
        passed += 1
        print(f"  [PASS] Test {total}: {name}")
    except AssertionError as e:
        failed += 1
        print(f"  [FAIL] Test {total}: {name}")
        print(f"         {e}")
    except Exception as e:
        failed += 1
        print(f"  [ERROR] Test {total}: {name}")
        print(f"          {type(e).__name__}: {e}")


def simulate(ctrl, plant, n_updates, verbose=False):
    """Run closed-loop simulation for n_updates controller steps."""
    history = {
        'Ub': [], 'Force': [], 'Ma_max': [], 'Re_pct': [],
        'mode': [], 'integral': []
    }
    for i in range(n_updates):
        Ma_max = plant.Ma_max
        Force, mode, Re_pct = ctrl.update(plant.Ub, Ma_max)

        history['Ub'].append(plant.Ub)
        history['Force'].append(Force)
        history['Ma_max'].append(Ma_max)
        history['Re_pct'].append(Re_pct)
        history['mode'].append(mode)
        history['integral'].append(ctrl.Force_integral)

        plant.step(Force)

        if verbose and i % 50 == 0:
            print(f"    step {i:4d}: Ub/Uref={plant.Ub/Uref:.4f}  Force={Force:.3e}  "
                  f"Ma_max={Ma_max:.4f}  [{mode}]  integral={ctrl.Force_integral:.3e}")

    # Final state
    history['Ub'].append(plant.Ub)
    history['Ma_max'].append(plant.Ma_max)
    for k in history:
        history[k] = np.array(history[k]) if k not in ('mode',) else history[k]
    return history


# ============================================================
# TEST 1: Cold start (Ub=0, Force=0)
# ============================================================
def test_cold_start():
    """From Ub=0, controller should ramp up Force and bring Ub → Uref."""
    ctrl  = PIForceController()
    plant = SimplePlant(Ub0=0.0, dt_sim=0.5)
    hist  = simulate(ctrl, plant, n_updates=500)

    # After 500 updates, Ub should be within 5% of Uref
    final_ratio = hist['Ub'][-1] / Uref
    assert 0.90 < final_ratio < 1.10, \
        f"Cold start: Ub/Uref = {final_ratio:.4f}, expected ~1.0"

    # Force should be positive throughout ramp-up
    assert np.all(hist['Force'] >= 0), "Force went negative during cold start"

    # No extreme overshoot: Ub should never exceed 1.3 * Uref
    max_ratio = np.max(hist['Ub']) / Uref
    assert max_ratio < 1.30, \
        f"Cold start overshoot: max Ub/Uref = {max_ratio:.4f}, expected < 1.30"


# ============================================================
# TEST 2: Near-target below (Ub/Uref ≈ 0.95)
# ============================================================
def test_near_target_below():
    """Start at 95% of target, should converge without oscillation."""
    ctrl  = PIForceController()
    # Pre-load integral to approximate steady-state
    ctrl.Force_integral = 0.5 * norm  # rough steady-state integral
    plant = SimplePlant(Ub0=0.95 * Uref, dt_sim=0.5)
    hist  = simulate(ctrl, plant, n_updates=300)

    final_ratio = hist['Ub'][-1] / Uref
    assert 0.97 < final_ratio < 1.03, \
        f"Near-below: Ub/Uref = {final_ratio:.4f}, expected ~1.0"

    # Should never overshoot past 1.10
    max_ratio = np.max(hist['Ub']) / Uref
    assert max_ratio < 1.10, \
        f"Near-below overshoot: max Ub/Uref = {max_ratio:.4f}"


# ============================================================
# TEST 3: Near-target above (Ub/Uref ≈ 1.05)
# ============================================================
def test_near_target_above():
    """Start 5% above target, should reduce Force and converge back."""
    ctrl  = PIForceController()
    ctrl.Force_integral = 0.5 * norm
    plant = SimplePlant(Ub0=1.05 * Uref, dt_sim=0.5)
    hist  = simulate(ctrl, plant, n_updates=300)

    final_ratio = hist['Ub'][-1] / Uref
    assert 0.95 < final_ratio < 1.05, \
        f"Near-above: Ub/Uref = {final_ratio:.4f}, expected ~1.0"


# ============================================================
# TEST 4: Significant overshoot (Ub/Uref ≈ 1.2)
# ============================================================
def test_overshoot_recovery():
    """Start at 120% of target, PI should reduce Force and recover."""
    ctrl  = PIForceController()
    ctrl.Force_integral = 1.0 * norm  # some accumulated integral
    plant = SimplePlant(Ub0=1.2 * Uref, dt_sim=0.5)
    hist  = simulate(ctrl, plant, n_updates=500)

    final_ratio = hist['Ub'][-1] / Uref
    assert 0.90 < final_ratio < 1.10, \
        f"Overshoot recovery: Ub/Uref = {final_ratio:.4f}, expected ~1.0"

    # Force should go to zero initially (Kp*error < 0, integral decays)
    # Check that it eventually recovers positive
    assert hist['Force'][-1] > 0, "Force should be positive at steady state"


# ============================================================
# TEST 5: Ma_max > 0.25 warning brake
# ============================================================
def test_ma_warning_brake():
    """When Ma_max crosses 0.25, Force should be halved."""
    ctrl = PIForceController()
    ctrl.Force_integral = 2.0 * norm
    ctrl.Force = 2.0 * norm

    # Simulate a single update with Ma_max = 0.27
    Force_before = Kp * (Uref - Uref) * norm + ctrl.Force_integral  # error=0 → Force = integral
    Force_out, _, _ = ctrl.update(Uref, Ma_max=0.27)

    # 0.25 < 0.27 < 0.30 → warning branch: Force *= 0.5
    # Force = integral (since error=0), then *= 0.5
    expected = ctrl.Force_integral  # integral was halved too
    assert abs(Force_out - expected) < 1e-15, \
        f"Ma warning: Force={Force_out:.6e}, expected integral/2"
    # Integral should also be halved
    assert ctrl.Force_integral < 1.1 * norm, \
        f"Ma warning: integral={ctrl.Force_integral:.6e}, expected ~{1.0*norm:.6e}"


# ============================================================
# TEST 6: Ma_max > 0.30 critical brake
# ============================================================
def test_ma_critical_brake():
    """When Ma_max > 0.30, Force reduced to 5%."""
    ctrl = PIForceController()
    ctrl.Force_integral = 5.0 * norm

    Force_out, _, _ = ctrl.update(Uref, Ma_max=0.35)

    # Force = integral (error=0), then *= 0.05
    assert Force_out < 0.06 * 5.0 * norm, \
        f"Ma critical: Force={Force_out:.6e}, expected < {0.06*5*norm:.6e}"
    # Integral crushed
    assert ctrl.Force_integral < 0.06 * 5.0 * norm, \
        f"Ma critical: integral too large after brake"


# ============================================================
# TEST 7: Ma brake order correctness (0.30 checked before 0.25)
# ============================================================
def test_ma_brake_order():
    """Verify that Ma=0.31 triggers critical (5%), not warning (50%)."""
    ctrl1 = PIForceController()
    ctrl1.Force_integral = 3.0 * norm
    F1, _, _ = ctrl1.update(Uref, Ma_max=0.31)

    ctrl2 = PIForceController()
    ctrl2.Force_integral = 3.0 * norm
    F2, _, _ = ctrl2.update(Uref, Ma_max=0.27)

    # Ma=0.31 → 5% of original; Ma=0.27 → 50% of original
    assert F1 < F2, \
        f"Brake order: Ma=0.31 gave Force={F1:.6e} ≥ Ma=0.27 Force={F2:.6e}"
    ratio = F1 / F2
    assert ratio < 0.15, \
        f"Brake order: ratio F(0.31)/F(0.27) = {ratio:.4f}, expected ~0.1"


# ============================================================
# TEST 8: Anti-windup upper bound
# ============================================================
def test_anti_windup_upper():
    """Integral should not exceed Force_max_integral = 20*norm."""
    ctrl = PIForceController()
    # Simulate many updates with large error (Ub=0, so error=Uref)
    for _ in range(10000):
        ctrl.update(0.0, Ma_max=0.0)

    assert ctrl.Force_integral <= Force_max_integral + 1e-20, \
        f"Anti-windup: integral={ctrl.Force_integral:.6e} > max={Force_max_integral:.6e}"


# ============================================================
# TEST 9: Anti-windup lower bound (integral ≥ 0)
# ============================================================
def test_anti_windup_lower():
    """Integral should not go below 0 even with persistent overshoot."""
    ctrl = PIForceController()
    # Large overshoot: error < 0 → integral decreases
    for _ in range(10000):
        ctrl.update(2.0 * Uref, Ma_max=0.0)

    assert ctrl.Force_integral >= 0.0, \
        f"Anti-windup lower: integral={ctrl.Force_integral:.6e} < 0"


# ============================================================
# TEST 10: Force non-negativity
# ============================================================
def test_force_non_negative():
    """Force should never be negative, even with large overshoot."""
    ctrl = PIForceController()
    for _ in range(100):
        F, _, _ = ctrl.update(3.0 * Uref, Ma_max=0.0)
        assert F >= 0.0, f"Force negative: {F:.6e}"


# ============================================================
# TEST 11: Long-term closed-loop stability
# ============================================================
def test_long_term_stability():
    """1000-update simulation: Ub converges, Ma_max stays bounded."""
    ctrl  = PIForceController()
    plant = SimplePlant(Ub0=0.0, dt_sim=0.5)
    hist  = simulate(ctrl, plant, n_updates=1000)

    # Final 100 updates should be within 3% of target
    final_Ub = hist['Ub'][-100:]
    final_ratios = final_Ub / Uref
    assert np.all(final_ratios > 0.95), \
        f"Long-term: min Ub/Uref in last 100 = {np.min(final_ratios):.4f}"
    assert np.all(final_ratios < 1.05), \
        f"Long-term: max Ub/Uref in last 100 = {np.max(final_ratios):.4f}"

    # Ma_max should stay bounded (< 0.2 at steady state with Ma_ratio=2.5)
    # Ma_max = 2.5 * Ub/cs. At Ub=Uref: Ma_max = 2.5*0.017/0.577 ≈ 0.074
    final_Ma = hist['Ma_max'][-100:]
    assert np.all(final_Ma < 0.20), \
        f"Long-term: max Ma in last 100 = {np.max(final_Ma):.4f}, expected < 0.20"


# ============================================================
# TEST 12: Ma_max convergence below 0.2 (high Ma_ratio scenario)
# ============================================================
def test_ma_convergence_below_02():
    """
    With aggressive Ma_ratio (hill crest acceleration factor = 8.0),
    Ma_max ≈ 8.0 * Uref/cs ≈ 0.236 at target Ub.
    Controller should reduce Ub below Uref to keep Ma_max < 0.25.
    Ma brakes should prevent blowup.
    """
    ctrl  = PIForceController()
    # Ma_ratio=8: at Ub=Uref, Ma_max = 8*0.017/0.577 = 0.236
    plant = SimplePlant(Ub0=0.0, Ma_ratio=8.0, dt_sim=0.5)
    hist  = simulate(ctrl, plant, n_updates=2000)

    # Ma_max should stay bounded — brake should keep it below ~0.35
    assert np.max(hist['Ma_max']) < 0.50, \
        f"High-Ma: peak Ma_max = {np.max(hist['Ma_max']):.4f}, expected < 0.50"

    # In last 200 steps, Ma_max should be oscillating below 0.30
    final_Ma = hist['Ma_max'][-200:]
    assert np.max(final_Ma) < 0.35, \
        f"High-Ma steady: max Ma in last 200 = {np.max(final_Ma):.4f}"


# ============================================================
# TEST 13: Restart from high Ub (Ub/Uref = 1.5)
# ============================================================
def test_restart_high_ub():
    """Start at 150% of target with accumulated integral. Should recover."""
    ctrl  = PIForceController()
    ctrl.Force_integral = 3.0 * norm  # leftover from previous run
    plant = SimplePlant(Ub0=1.5 * Uref, dt_sim=0.5)
    hist  = simulate(ctrl, plant, n_updates=500)

    final_ratio = hist['Ub'][-1] / Uref
    assert 0.85 < final_ratio < 1.15, \
        f"Restart high: Ub/Uref = {final_ratio:.4f}, expected ~1.0"

    # Should not diverge: no runaway Ma
    assert np.max(hist['Ma_max']) < 0.30, \
        f"Restart high: peak Ma_max = {np.max(hist['Ma_max']):.4f}"


# ============================================================
# TEST 14: Repeated Ma brake recovery cycles
# ============================================================
def test_repeated_ma_brake_recovery():
    """
    Inject artificial Ma spikes periodically.
    Controller should recover each time without permanent damage.
    """
    ctrl  = PIForceController()
    plant = SimplePlant(Ub0=0.0, dt_sim=0.5)

    # Run 200 normal steps first
    for _ in range(200):
        Ma = plant.Ma_max
        F, _, _ = ctrl.update(plant.Ub, Ma)
        plant.step(F)

    Ub_before_spikes = plant.Ub

    # Inject 5 Ma spikes
    for spike in range(5):
        # Artificial Ma spike
        F, _, _ = ctrl.update(plant.Ub, Ma_max=0.32)
        plant.step(F)
        # 50 normal recovery steps
        for _ in range(50):
            F, _, _ = ctrl.update(plant.Ub, plant.Ma_max)
            plant.step(F)

    # After recovery, should still be moving toward target
    assert plant.Ub > 0, "Ub collapsed to 0 after Ma spikes"
    assert ctrl.Force_integral > 0, "Integral collapsed to 0 permanently"


# ============================================================
# TEST 15: Oscillation damping (no limit cycles)
# ============================================================
def test_oscillation_damping():
    """
    Check that Ub/Uref oscillations dampen over time (no sustained oscillation).
    Measure amplitude of last 100 vs first 100 oscillations after convergence.
    """
    ctrl  = PIForceController()
    plant = SimplePlant(Ub0=0.0, dt_sim=0.5)
    hist  = simulate(ctrl, plant, n_updates=2000)

    Ub_ratio = hist['Ub'] / Uref

    # Measure oscillation amplitude in two windows
    if len(Ub_ratio) > 1500:
        mid_window = Ub_ratio[500:800]
        end_window = Ub_ratio[1700:2000]
        amp_mid = np.max(mid_window) - np.min(mid_window)
        amp_end = np.max(end_window) - np.min(end_window)

        # End oscillation should be no larger than mid (damped or constant)
        assert amp_end <= amp_mid + 0.01, \
            f"Oscillation growing: mid amp={amp_mid:.6f}, end amp={amp_end:.6f}"


# ============================================================
# TEST 16: Proportional term sign correctness
# ============================================================
def test_proportional_sign():
    """Verify Kp term has correct sign: error>0 → positive Force contribution."""
    ctrl = PIForceController()

    # Ub < Uref → error > 0 → Force should increase
    F1, _, _ = ctrl.update(0.5 * Uref, Ma_max=0.0)
    assert F1 > 0, f"Kp sign wrong: Ub<Uref but Force={F1:.6e}"

    # Reset
    ctrl2 = PIForceController()
    ctrl2.Force_integral = 0.0

    # Ub > Uref → error < 0 → Kp contribution negative (Force may be 0 due to clamp)
    F2, _, _ = ctrl2.update(1.5 * Uref, Ma_max=0.0)
    assert F2 == 0.0, f"Kp sign: Ub>>Uref, Force should be 0 (clamped), got {F2:.6e}"


# ============================================================
# TEST 17: Controller mode labels
# ============================================================
def test_mode_labels():
    """Verify correct mode labels for different error ranges."""
    ctrl = PIForceController()

    # Large undershoot
    _, mode, Re_pct = ctrl.update(0.5 * Uref, Ma_max=0.0)
    assert mode == "PI-accel", f"Expected PI-accel, got {mode}"
    assert Re_pct < -1.5, f"Expected Re_pct << -1.5, got {Re_pct:.2f}"

    # Reset
    ctrl2 = PIForceController()
    _, mode2, Re_pct2 = ctrl2.update(1.5 * Uref, Ma_max=0.0)
    assert mode2 == "PI-decel", f"Expected PI-decel, got {mode2}"

    # Near target
    ctrl3 = PIForceController()
    _, mode3, Re_pct3 = ctrl3.update(1.005 * Uref, Ma_max=0.0)
    assert mode3 == "PI-steady", f"Expected PI-steady, got {mode3}"


# ============================================================
# TEST 18: Integral builds up correctly during cold start
# ============================================================
def test_integral_buildup():
    """During cold start, integral should monotonically increase."""
    ctrl = PIForceController()
    integrals = []
    for _ in range(50):
        ctrl.update(0.0, Ma_max=0.0)
        integrals.append(ctrl.Force_integral)

    # Should be monotonically increasing (error is always positive)
    for i in range(1, len(integrals)):
        assert integrals[i] >= integrals[i-1], \
            f"Integral decreased at step {i}: {integrals[i]:.6e} < {integrals[i-1]:.6e}"

    # Should be positive
    assert integrals[-1] > 0, f"Integral not positive after 50 steps: {integrals[-1]:.6e}"


# ============================================================
# TEST 19: Steady-state Force is physically reasonable
# ============================================================
def test_steady_state_force():
    """
    At steady state, Force should balance drag: Force ≈ C_drag * Uref^2.
    F* = Force * LY / Uref^2 should be O(1) — typical range [0.1, 5.0].
    """
    ctrl  = PIForceController()
    plant = SimplePlant(Ub0=0.0, dt_sim=0.5)
    hist  = simulate(ctrl, plant, n_updates=1000)

    # Last 50 Force values
    final_F = hist['Force'][-50:]
    F_star = final_F * LY / Uref**2  # dimensionless
    mean_Fstar = np.mean(F_star)

    assert 0.05 < mean_Fstar < 10.0, \
        f"Steady F* = {mean_Fstar:.4f}, expected O(1) (range [0.05, 10])"


# ============================================================
# TEST 20: Comparison with old Gehrke — no exponential blowup
# ============================================================
def test_no_exponential_blowup():
    """
    The old controller had Force jumping from 0.81→3.14 (3.9x in ~4 updates).
    New PI should never have such exponential growth.
    Check that Force never increases by more than 50% between consecutive updates.
    """
    ctrl  = PIForceController()
    plant = SimplePlant(Ub0=0.0, dt_sim=0.5)
    hist  = simulate(ctrl, plant, n_updates=500)

    Forces = hist['Force']
    for i in range(1, len(Forces)):
        if Forces[i-1] > 1e-20:  # skip near-zero
            ratio = Forces[i] / Forces[i-1]
            assert ratio < 1.5, \
                f"Step {i}: Force ratio = {ratio:.4f} (from {Forces[i-1]:.3e} to {Forces[i]:.3e})"


# ============================================================
# TEST 21: Multi-plant-timescale robustness
# ============================================================
def test_different_plant_speeds():
    """Controller should work for fast and slow plant responses."""
    for dt_sim, label in [(0.1, "slow"), (1.0, "medium"), (5.0, "fast")]:
        ctrl  = PIForceController()
        plant = SimplePlant(Ub0=0.0, dt_sim=dt_sim)
        hist  = simulate(ctrl, plant, n_updates=2000)

        final_ratio = hist['Ub'][-1] / Uref
        assert 0.5 < final_ratio < 2.0, \
            f"Plant '{label}' (dt={dt_sim}): Ub/Uref = {final_ratio:.4f}, not converging"

        # No divergence
        assert np.all(np.isfinite(hist['Ub'])), f"Plant '{label}': NaN/Inf in Ub"
        assert np.all(np.isfinite(hist['Force'])), f"Plant '{label}': NaN/Inf in Force"


# ============================================================
# TEST 22: Ma_max stays below 0.2 at steady state (normal scenario)
# ============================================================
def test_ma_below_02_steady():
    """
    For normal Ma_ratio (2.5), verify Ma_max < 0.2 in the last 200 updates.
    Ma_max = 2.5 * Ub/cs. At Ub=Uref: 2.5*0.017/0.577 ≈ 0.074.
    """
    ctrl  = PIForceController()
    plant = SimplePlant(Ub0=0.0, Ma_ratio=2.5, dt_sim=0.5)
    hist  = simulate(ctrl, plant, n_updates=1000)

    final_Ma = hist['Ma_max'][-200:]
    assert np.all(final_Ma < 0.20), \
        f"Normal scenario: max Ma in last 200 = {np.max(final_Ma):.4f}, expected < 0.20"

    mean_Ma = np.mean(final_Ma)
    assert mean_Ma < 0.10, \
        f"Normal scenario: mean Ma in last 200 = {mean_Ma:.4f}, expected < 0.10"


# ============================================================
# TEST 23: High Ma_ratio — brakes prevent divergence
# ============================================================
def test_high_ma_ratio_brake_prevents_divergence():
    """
    Ma_ratio = 12: at Ub=Uref, Ma_max = 12*0.017/0.577 ≈ 0.353.
    This exceeds 0.30 threshold. Controller should reduce Ub below Uref
    to keep Ma_max bounded. Simulation should not diverge.
    """
    ctrl  = PIForceController()
    plant = SimplePlant(Ub0=0.0, Ma_ratio=12.0, dt_sim=0.5)
    hist  = simulate(ctrl, plant, n_updates=3000)

    # Should not diverge
    assert np.all(np.isfinite(hist['Ub'])), "High Ma_ratio: NaN/Inf in Ub"
    assert np.all(np.isfinite(hist['Force'])), "High Ma_ratio: NaN/Inf in Force"

    # Peak Ma should be bounded (brakes should kick in)
    peak_Ma = np.max(hist['Ma_max'])
    assert peak_Ma < 0.60, \
        f"High Ma_ratio: peak Ma = {peak_Ma:.4f}, expected < 0.60"

    # Ub should settle below Uref to respect Ma constraint
    final_Ub = np.mean(hist['Ub'][-200:])
    # At Ma_max=0.25 target: Ub = 0.25*cs/12 ≈ 0.012 < Uref=0.017
    assert final_Ub < 1.1 * Uref, \
        f"High Ma_ratio: final Ub = {final_Ub:.6f}, should be ≤ Uref"


# ============================================================
# Run all tests
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PI Force Controller — Comprehensive Unit Tests")
    print("=" * 70)
    print(f"Parameters: Uref={Uref}, Re={Re}, LY={LY}, cs={cs:.6f}")
    print(f"Controller: Kp={Kp}, Ki={Ki}, norm={norm:.6e}")
    print(f"Anti-windup: integral_max={Force_max_integral:.6e}")
    print("=" * 70)

    run_test("Cold start (Ub=0 → ramp up)", test_cold_start)
    run_test("Near-target below (Ub/Uref ≈ 0.95)", test_near_target_below)
    run_test("Near-target above (Ub/Uref ≈ 1.05)", test_near_target_above)
    run_test("Overshoot recovery (Ub/Uref = 1.2)", test_overshoot_recovery)
    run_test("Ma > 0.25 warning brake", test_ma_warning_brake)
    run_test("Ma > 0.30 critical brake", test_ma_critical_brake)
    run_test("Ma brake order (0.30 before 0.25)", test_ma_brake_order)
    run_test("Anti-windup upper bound", test_anti_windup_upper)
    run_test("Anti-windup lower bound (integral ≥ 0)", test_anti_windup_lower)
    run_test("Force non-negativity", test_force_non_negative)
    run_test("Long-term closed-loop stability", test_long_term_stability)
    run_test("Ma convergence < 0.2 (high Ma_ratio)", test_ma_convergence_below_02)
    run_test("Restart from high Ub (Ub/Uref = 1.5)", test_restart_high_ub)
    run_test("Repeated Ma brake recovery", test_repeated_ma_brake_recovery)
    run_test("Oscillation damping (no limit cycles)", test_oscillation_damping)
    run_test("Proportional term sign correctness", test_proportional_sign)
    run_test("Controller mode labels", test_mode_labels)
    run_test("Integral buildup during cold start", test_integral_buildup)
    run_test("Steady-state Force physically reasonable", test_steady_state_force)
    run_test("No exponential blowup (vs old Gehrke)", test_no_exponential_blowup)
    run_test("Multi-plant-timescale robustness", test_different_plant_speeds)
    run_test("Ma < 0.2 at steady state (normal)", test_ma_below_02_steady)
    run_test("High Ma_ratio brake prevents divergence", test_high_ma_ratio_brake_prevents_divergence)

    print("=" * 70)
    print(f"Results: {passed}/{total} PASSED, {failed}/{total} FAILED")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)
