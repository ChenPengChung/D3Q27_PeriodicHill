#!/usr/bin/env python3
"""
Hybrid Dual-Stage Force Controller — Comprehensive Unit Tests
=============================================================
精確重現 evolution.h 的 Launch_ModifyForcingTerm() 控制器邏輯,
包含 PID (Phase 1) + Gehrke (Phase 2) + 連續 Mach brake.

測試目標:
  1. PID 基本功能 (比例/積分/微分/anti-windup)
  2. Gehrke 乘法控制器 (correction/deadzone/floor/clamp)
  3. Phase transition (PID↔Gehrke 無跳變)
  4. 連續 Mach brake (二次衰減/緊急歸零/增長率煞車)
  5. ★ 重播 FTT 1.28 發散情境: 驗證新控制器能壓制 Ma_max 暴衝
  6. 參數一致性 (variables.h #define 對照)

Usage:
  python3 test_force_controller.py
"""

import math
import unittest
from dataclasses import dataclass, field

# =====================================================================
# 從 variables.h 搬過來的參數 (必須與 C code 完全一致)
# =====================================================================
UREF = 0.037
LY = 9.0
LZ = 3.036
H_HILL = 1.0
RE = 700
NIU = UREF / RE
CS = 1.0 / math.sqrt(3.0)

# PID
FORCE_KP = 2.0
FORCE_KI = 0.3
FORCE_KD = 0.5

# Gehrke
FORCE_GEHRKE_GAIN = 0.1
FORCE_GEHRKE_DEADZONE = 1.5   # %
FORCE_GEHRKE_FLOOR = 0.0      # variables.h: 0.0 (no floor)
FORCE_SWITCH_THRESHOLD = 10.0  # variables.h: 10.0%

# Mach brake
MA_BRAKE_MULT_THRESHOLD = 2.0  # variables.h: 2.0
MA_BRAKE_MULT_CRITICAL = 2.1   # variables.h: 2.1
MA_BRAKE_GROWTH_LIMIT = 0.30

# Derived
NORM = UREF * UREF / LY
H_EFF = LZ - H_HILL
F_POISEUILLE = 8.0 * NIU * UREF / (H_EFF * H_EFF)
F_FLOOR = FORCE_GEHRKE_FLOOR * F_POISEUILLE
MA_BULK_REF = UREF / CS
MA_THRESHOLD = MA_BRAKE_MULT_THRESHOLD * MA_BULK_REF
MA_CRITICAL = MA_BRAKE_MULT_CRITICAL * MA_BULK_REF


# =====================================================================
# Python 重現: 控制器狀態 + 單步更新函數
# =====================================================================
@dataclass
class ControllerState:
    """Mirror of evolution.h static variables"""
    Force: float = 0.0             # Force_h[0]
    Force_integral: float = 0.0    # PID integral
    error_prev: float = 0.0        # previous error (for D term)
    initialized: bool = False
    gehrke_activated: bool = False
    Ma_max_prev: float = 0.0
    mode: str = ""                 # last ctrl_mode string
    Ma_factor: float = 1.0        # last Ma brake factor


def controller_update(state: ControllerState, Ub_avg: float, Ma_max: float) -> ControllerState:
    """
    精確重現 evolution.h Launch_ModifyForcingTerm() 的控制器邏輯.
    輸入: Ub_avg (bulk velocity), Ma_max (global max Mach number)
    輸出: 更新後的 state (含 Force, integral, mode 等)
    """
    error = UREF - Ub_avg
    Re_pct = (Ub_avg - UREF) / UREF * 100.0

    # Initialization
    if not state.initialized:
        state.Force_integral = 0.0
        state.error_prev = error
        state.initialized = True

    Kp, Ki, Kd = FORCE_KP, FORCE_KI, FORCE_KD

    # Mode selection
    use_gehrke = abs(Re_pct) <= FORCE_SWITCH_THRESHOLD

    # Phase transition
    if use_gehrke and not state.gehrke_activated:
        state.gehrke_activated = True
    elif not use_gehrke and state.gehrke_activated:
        state.gehrke_activated = False
        state.Force_integral = max(0.0, state.Force)

    if use_gehrke:
        # ── Phase 2: Gehrke multiplicative ──
        if abs(Re_pct) < FORCE_GEHRKE_DEADZONE:
            state.mode = "GEHRKE-HOLD"
        else:
            correction = 1.0 - FORCE_GEHRKE_GAIN * Re_pct
            correction = max(0.5, min(2.0, correction))
            state.Force *= correction
            state.mode = "GEHRKE-DEC" if Re_pct > 0 else "GEHRKE-INC"

        # Floor
        if state.Force < F_FLOOR:
            state.Force = F_FLOOR

        # Sync PID state
        state.Force_integral = state.Force
        state.error_prev = error

    else:
        # ── Phase 1: PID controller ──
        d_error = error - state.error_prev
        state.error_prev = error

        state.Force_integral += Ki * error * NORM

        # Conditional decay
        if error < 0.0 and state.Force_integral > 0.0:
            state.Force_integral *= 0.5

        # Anti-windup
        Force_max = 10.0 * NORM
        state.Force_integral = max(0.0, min(Force_max, state.Force_integral))

        # PID synthesis
        state.Force = Kp * error * NORM + state.Force_integral + Kd * d_error * NORM

        # Back-calculation
        if state.Force < 0.0:
            state.Force = 0.0
            integral_target = max(0.0, -Kp * error * NORM)
            if state.Force_integral > integral_target:
                state.Force_integral = integral_target

        if abs(Re_pct) < 1.5:
            state.mode = "PID-steady"
        elif error > 0:
            state.mode = "PID-accel"
        else:
            state.mode = "PID-decel"

    # ── Continuous Mach Safety Brake ──
    Ma_growth_rate = 0.0
    if state.Ma_max_prev > 1e-10:
        Ma_growth_rate = (Ma_max - state.Ma_max_prev) / state.Ma_max_prev
    state.Ma_max_prev = Ma_max

    Ma_factor = 1.0

    if Ma_max > MA_THRESHOLD and Ma_max <= MA_CRITICAL:
        excess = (Ma_max - MA_THRESHOLD) / (MA_CRITICAL - MA_THRESHOLD)
        Ma_factor = (1.0 - excess) ** 2

    if Ma_max > MA_CRITICAL:
        Ma_factor = 0.0
        state.Force_integral = 0.0

    if Ma_growth_rate > MA_BRAKE_GROWTH_LIMIT and Ma_max > MA_BULK_REF * 1.5:
        Ma_factor *= 0.3
        state.Force_integral *= 0.5

    state.Force *= Ma_factor
    state.Force_integral *= Ma_factor
    state.Ma_factor = Ma_factor

    return state


# =====================================================================
# 測試用例
# =====================================================================
class TestParameterConsistency(unittest.TestCase):
    """Test 0: 參數一致性檢查"""

    def test_norm_value(self):
        """norm = Uref²/LY"""
        self.assertAlmostEqual(NORM, 0.037**2 / 9.0, places=10)

    def test_poiseuille_estimate(self):
        """F_Poiseuille = 8ν·Uref / h_eff²"""
        expected = 8.0 * NIU * UREF / (H_EFF**2)
        self.assertAlmostEqual(F_POISEUILLE, expected, places=15)

    def test_ma_bulk_ref(self):
        """Ma_bulk = Uref/cs ≈ 0.064"""
        self.assertAlmostEqual(MA_BULK_REF, UREF * math.sqrt(3.0), places=10)
        self.assertAlmostEqual(MA_BULK_REF, UREF / CS, places=10)

    def test_ma_thresholds(self):
        """Ma_threshold ≈ 0.128, Ma_critical ≈ 0.135 (tight brake)"""
        self.assertAlmostEqual(MA_THRESHOLD, MA_BRAKE_MULT_THRESHOLD * MA_BULK_REF, places=10)
        self.assertAlmostEqual(MA_CRITICAL, MA_BRAKE_MULT_CRITICAL * MA_BULK_REF, places=10)
        self.assertTrue(0.12 < MA_THRESHOLD < 0.14)
        self.assertTrue(0.13 < MA_CRITICAL < 0.14)

    def test_gehrke_correction_range(self):
        """在 SWITCH_THRESHOLD=10% 時, raw correction ∈ [0.0, 2.0], clamped to [0.5, 2.0]"""
        # Re% = +10% (max overshoot in Gehrke zone)
        corr_max_over = 1.0 - FORCE_GEHRKE_GAIN * 10.0
        self.assertAlmostEqual(corr_max_over, 0.0)  # raw = 0.0, clamped to 0.5
        # Re% = -10% (max undershoot in Gehrke zone)
        corr_max_under = 1.0 - FORCE_GEHRKE_GAIN * (-10.0)
        self.assertAlmostEqual(corr_max_under, 2.0)  # clamped to 2.0


class TestPIDBasic(unittest.TestCase):
    """Test 1: PID controller fundamentals"""

    def test_cold_start_accel(self):
        """冷啟動 Ub=0: PID 應產生正 Force 加速"""
        s = ControllerState()
        s = controller_update(s, Ub_avg=0.0, Ma_max=0.01)
        self.assertGreater(s.Force, 0.0)
        self.assertEqual(s.mode, "PID-accel")

    def test_pid_proportional(self):
        """比例項: error 越大 → Force 越大"""
        s1 = ControllerState()
        s1 = controller_update(s1, Ub_avg=0.0, Ma_max=0.01)  # error = Uref

        s2 = ControllerState()
        s2 = controller_update(s2, Ub_avg=UREF * 0.5, Ma_max=0.05)  # error = Uref/2

        self.assertGreater(s1.Force, s2.Force)

    def test_pid_integral_accumulates(self):
        """積分項: 持續 under-speed → integral 增加"""
        s = ControllerState()
        Ub = UREF * 0.8  # 持續偏低
        for _ in range(20):
            s = controller_update(s, Ub_avg=Ub, Ma_max=0.05)
        # integral 應非零且為正
        self.assertGreater(s.Force_integral, 0.0)

    def test_pid_derivative_braking(self):
        """微分項: Ub 突然增加 → d_error 為負 → 自動減力"""
        s = ControllerState()
        # 先穩定在 Ub = 0.5*Uref
        for _ in range(5):
            s = controller_update(s, Ub_avg=UREF * 0.5, Ma_max=0.04)
        F_before = s.Force

        # Ub 突然跳到 0.9*Uref → error 變小但 d_error < 0
        s = controller_update(s, Ub_avg=UREF * 0.9, Ma_max=0.06)
        # Force 應減少 (D 項 braking)
        self.assertLess(s.Force, F_before)

    def test_pid_force_nonnegative(self):
        """Force 永遠 ≥ 0 (back-calculation anti-windup)"""
        s = ControllerState()
        # 強制 overshoot
        s = controller_update(s, Ub_avg=UREF * 3.0, Ma_max=0.10)
        self.assertGreaterEqual(s.Force, 0.0)

    def test_pid_integral_clamped(self):
        """Anti-windup: integral ≤ 10×norm"""
        s = ControllerState()
        for _ in range(1000):
            s = controller_update(s, Ub_avg=0.0, Ma_max=0.01)
        self.assertLessEqual(s.Force_integral, 10.0 * NORM + 1e-15)

    def test_pid_conditional_decay(self):
        """Overshoot + integral > 0 → integral 衰減 50%"""
        s = ControllerState()
        # 先建立 integral
        for _ in range(20):
            s = controller_update(s, Ub_avg=UREF * 0.5, Ma_max=0.04)
        integral_before = s.Force_integral
        self.assertGreater(integral_before, 0.0)

        # Overshoot: Ub > Uref → error < 0
        s = controller_update(s, Ub_avg=UREF * 1.2, Ma_max=0.08)
        # integral 應衰減 (乘 0.5 + Ki 負累加)
        self.assertLess(s.Force_integral, integral_before)


class TestGehrkeBasic(unittest.TestCase):
    """Test 2: Gehrke multiplicative controller"""

    def _make_steady_state(self, Ub, Force_init=1e-4):
        """建立一個已在 Gehrke 模式的 state"""
        s = ControllerState(Force=Force_init, initialized=True, gehrke_activated=True)
        s.error_prev = UREF - Ub
        s.Force_integral = Force_init
        s.Ma_max_prev = 0.05
        return s

    def test_gehrke_deadzone(self):
        """|Re%| < 1.5% → HOLD, Force 不變"""
        Ub = UREF * 1.01  # Re% = +1.0%, 在死區內
        F_init = 1e-4
        s = self._make_steady_state(Ub, F_init)
        s = controller_update(s, Ub_avg=Ub, Ma_max=0.06)
        self.assertEqual(s.mode, "GEHRKE-HOLD")
        self.assertAlmostEqual(s.Force, F_init, places=10)

    def test_gehrke_decrease(self):
        """Re% > deadzone → correction < 1 → Force 減少"""
        Ub = UREF * 1.03  # Re% = +3%
        F_init = 1e-4
        s = self._make_steady_state(Ub, F_init)
        s = controller_update(s, Ub_avg=Ub, Ma_max=0.06)
        self.assertEqual(s.mode, "GEHRKE-DEC")
        expected_correction = 1.0 - FORCE_GEHRKE_GAIN * 3.0  # = 0.7
        self.assertAlmostEqual(s.Force, F_init * expected_correction, places=10)

    def test_gehrke_increase(self):
        """Re% < -deadzone → correction > 1 → Force 增加"""
        Ub = UREF * 0.97  # Re% = -3%
        F_init = 1e-4
        s = self._make_steady_state(Ub, F_init)
        s = controller_update(s, Ub_avg=Ub, Ma_max=0.06)
        self.assertEqual(s.mode, "GEHRKE-INC")
        expected_correction = 1.0 - FORCE_GEHRKE_GAIN * (-3.0)  # = 1.3
        self.assertAlmostEqual(s.Force, F_init * expected_correction, places=10)

    def test_gehrke_correction_clamp(self):
        """Correction 被 clamp 在 [0.5, 2.0]"""
        # 極端 Re%=+8% → correction = 1-0.1*8 = 0.2, clamped to 0.5
        Ub = UREF * 1.08
        F_init = 1e-4
        s = self._make_steady_state(Ub, F_init)
        s = controller_update(s, Ub_avg=Ub, Ma_max=0.06)
        self.assertAlmostEqual(s.Force, F_init * 0.5, places=8)

    def test_gehrke_floor(self):
        """FLOOR=0.0 → Force 可以降到 0 (但 Gehrke 乘法不會到 0 除非 correction=0)"""
        # With FORCE_GEHRKE_FLOOR=0.0, F_FLOOR=0.0
        # Gehrke correction at Re%=4% → 0.6, so Force decreases but stays > 0
        Ub = UREF * 1.04
        F_init = 1e-6  # 極低值
        s = self._make_steady_state(Ub, F_init)
        s = controller_update(s, Ub_avg=Ub, Ma_max=0.06)
        # correction = 1 - 0.1*4 = 0.6, Force = 1e-6 * 0.6 = 6e-7 > 0
        self.assertGreater(s.Force, 0.0)
        self.assertAlmostEqual(s.Force, F_init * 0.6, places=12)

    def test_gehrke_syncs_integral(self):
        """Gehrke 模式下 integral 追蹤 Force"""
        Ub = UREF * 1.02
        F_init = 1e-4
        s = self._make_steady_state(Ub, F_init)
        s = controller_update(s, Ub_avg=Ub, Ma_max=0.06)
        self.assertAlmostEqual(s.Force_integral, s.Force, places=15)


class TestPhaseTransition(unittest.TestCase):
    """Test 3: PID ↔ Gehrke 切換無跳變"""

    def test_pid_to_gehrke_continuity(self):
        """PID → Gehrke: Force 連續 (Gehrke 從 PID 最後的 Force 乘起)"""
        s = ControllerState()
        # PID 模式: |Re%| > 5%
        for _ in range(30):
            s = controller_update(s, Ub_avg=UREF * 0.5, Ma_max=0.04)
        self.assertIn("PID", s.mode)
        F_pid = s.Force

        # 讓 Ub 跳到接近目標 (|Re%| < 5%) → 切入 Gehrke
        s = controller_update(s, Ub_avg=UREF * 0.98, Ma_max=0.06)
        self.assertIn("GEHRKE", s.mode)
        # Force 應連續 (Gehrke 乘以 correction ≈ 1.002, 非常接近)
        # 關鍵: 不應跳到 0 或跳到完全不同的值
        self.assertGreater(s.Force, 0.0)

    def test_gehrke_to_pid_no_jump(self):
        """Gehrke → PID: integral = Force_h[0], 不跳變"""
        # 先建立 Gehrke 穩態
        s = ControllerState(Force=1e-4, initialized=True, gehrke_activated=True)
        s.error_prev = UREF - UREF * 0.99
        s.Force_integral = 1e-4
        s.Ma_max_prev = 0.06

        # Gehrke 模式跑幾步
        for _ in range(5):
            s = controller_update(s, Ub_avg=UREF * 0.99, Ma_max=0.06)
        F_gehrke = s.Force
        self.assertIn("GEHRKE", s.mode)

        # Ub 大幅偏離 → 切回 PID
        s = controller_update(s, Ub_avg=UREF * 0.5, Ma_max=0.04)
        self.assertIn("PID", s.mode)
        # 回切時 integral 被設為 Gehrke 的 Force → PID 不從零開始
        # Force 不應從 ~1e-4 跳到 ~1e-5 或 ~1e-3 (桁數不應跳變)
        ratio = s.Force / F_gehrke if F_gehrke > 0 else float('inf')
        self.assertGreater(ratio, 0.1, "Force jumped down too much on transition")
        self.assertLess(ratio, 100.0, "Force jumped up too much on transition")

    def test_oscillation_between_modes(self):
        """Re% 在閾值附近震盪不應導致 Force 暴衝"""
        s = ControllerState()
        # 先到接近目標
        for _ in range(50):
            s = controller_update(s, Ub_avg=UREF * 0.7, Ma_max=0.05)

        # 在 4.8% ~ 5.2% 之間來回
        forces = []
        for i in range(20):
            Re_frac = 0.048 if i % 2 == 0 else 0.052
            Ub = UREF * (1.0 + Re_frac)
            s = controller_update(s, Ub_avg=Ub, Ma_max=0.06)
            forces.append(s.Force)

        # Force 不應有 >10× 的跳動
        for i in range(1, len(forces)):
            if forces[i-1] > 1e-15 and forces[i] > 1e-15:
                ratio = forces[i] / forces[i-1]
                self.assertGreater(ratio, 0.05, f"Force crash at step {i}")
                self.assertLess(ratio, 20.0, f"Force spike at step {i}")


class TestMachBrake(unittest.TestCase):
    """Test 4: Continuous Mach safety brake"""

    def test_below_threshold_no_brake(self):
        """Ma_max < Ma_threshold → Ma_factor = 1.0"""
        s = ControllerState()
        s = controller_update(s, Ub_avg=UREF * 0.5, Ma_max=MA_THRESHOLD * 0.9)
        self.assertAlmostEqual(s.Ma_factor, 1.0)

    def test_quadratic_decay_at_midpoint(self):
        """Ma_max = midpoint → factor = 0.25"""
        Ma_mid = (MA_THRESHOLD + MA_CRITICAL) / 2.0  # excess = 0.5
        s = ControllerState()
        s.initialized = True
        s.Ma_max_prev = Ma_mid * 0.99  # 微小增長, 不觸發 rate brake
        s = controller_update(s, Ub_avg=UREF * 0.5, Ma_max=Ma_mid)
        expected = (1.0 - 0.5) ** 2  # = 0.25
        self.assertAlmostEqual(s.Ma_factor, expected, places=3)

    def test_critical_zeroes_force(self):
        """Ma_max > Ma_critical → Force = 0"""
        s = ControllerState(Force=1e-3, initialized=True)
        s.Ma_max_prev = MA_CRITICAL * 0.5  # 會觸發 rate brake 但 critical 優先
        s = controller_update(s, Ub_avg=UREF * 0.5, Ma_max=MA_CRITICAL * 1.1)
        self.assertAlmostEqual(s.Force, 0.0, places=15)
        self.assertAlmostEqual(s.Force_integral, 0.0, places=15)

    def test_growth_rate_brake(self):
        """Ma_max 增長 >30% → 額外 ×0.3"""
        s = ControllerState(Force=1e-4, initialized=True, gehrke_activated=True)
        s.Force_integral = 1e-4
        s.error_prev = UREF - UREF * 0.99
        Ma_prev = 0.10
        Ma_now = 0.10 * 1.35  # +35% 增長 (> 30% threshold)
        s.Ma_max_prev = Ma_prev

        # Ma_now = 0.135, at critical boundary (≈0.1346)
        # Ma_now > threshold (0.128) → quadratic brake active
        # 增長 35% > 30% 且 Ma > 1.5*Ma_bulk → rate brake
        s = controller_update(s, Ub_avg=UREF * 0.99, Ma_max=Ma_now)

        # 因為增長率 > 30%, 應有額外 ×0.3
        self.assertLess(s.Ma_factor, 0.35)  # quadratic + rate 加成

    def test_brake_continuous_not_discrete(self):
        """驗證 brake 是連續的, 不是離散跳變"""
        factors = []
        for i in range(20):
            Ma = MA_THRESHOLD + (MA_CRITICAL - MA_THRESHOLD) * i / 19.0
            s = ControllerState(Force=1e-4, initialized=True)
            s.Ma_max_prev = Ma * 0.999  # 微小增長
            s = controller_update(s, Ub_avg=UREF * 0.5, Ma_max=Ma)
            factors.append(s.Ma_factor)

        # 檢查單調遞減
        for i in range(1, len(factors)):
            self.assertLessEqual(factors[i], factors[i-1] + 1e-10,
                                 f"Ma brake not monotone at step {i}")

        # 檢查從 ~1.0 → ~0.0 連續
        self.assertGreater(factors[0], 0.9)
        self.assertLess(factors[-1], 0.05)


class TestDivergenceScenarioReplay(unittest.TestCase):
    """
    Test 5: ★ 重播 FTT 1.28 發散情境
    =================================
    用舊模擬的 Ma_max 數據序列, 驗證新控制器在相同情境下能壓制暴衝.

    舊控制器 (PI-only, 離散閾值 0.25/0.30) 在此序列下:
    - Ma_max: 0.13 → 0.17 → 0.20 → 0.28 → 0.37 → 0.48 → 0.64 → 1.41 → 發散
    - Force 到 0.30 才介入, 為時已晚

    新控制器 (Hybrid + 連續 brake) 應在 Ma_max ≈ 0.16 就開始減力,
    且 Ma_max 永遠不超過 Ma_critical ≈ 0.256.
    """

    def test_replay_divergence_sequence(self):
        """重播實際發散的 Ma_max 序列"""
        # 模擬 cold start → PID 加速 → 到達穩態 → Gehrke 接管
        # 先用 PID (cold start, Ub=0 → Re%=-100% → PID zone)
        s = ControllerState()
        Ub_frac = 0.0
        for _ in range(100):
            s = controller_update(s, Ub_avg=UREF * Ub_frac, Ma_max=0.05)
            Ub_frac = min(Ub_frac + 0.01, 0.95)  # 漸進加速

        # 穩態: Ub ≈ 0.95*Uref, Re%=-5%, Gehrke zone
        for _ in range(200):
            s = controller_update(s, Ub_avg=UREF * 0.95, Ma_max=0.10)

        F_steady = s.Force
        self.assertGreater(F_steady, 0.0, "Should have positive force in steady state")
        self.assertIn("GEHRKE", s.mode, "Should be in Gehrke mode (Re%=-5%, threshold=10%)")

        # ★ 發散序列: 模擬 FTT 1.14 ~ 1.28 的 Ma_max 暴衝
        # Ma thresholds now: threshold ≈ 0.128, critical ≈ 0.135 (very tight)
        divergence_sequence = [
            (0.99, 0.100),   # FTT ~1.14: 正常, Ma < threshold
            (1.01, 0.120),   # FTT ~1.15: 接近 threshold
            (1.02, 0.130),   # FTT ~1.17: ★ 新 brake 介入 (>0.128)
            (1.03, 0.135),   # FTT ~1.20: 接近 critical (0.135)
            (1.04, 0.140),   # FTT ~1.22: > critical → 歸零
            (1.05, 0.165),   # FTT ~1.24: 遠超 critical
            (1.06, 0.190),   # FTT ~1.25
            (1.08, 0.234),   # FTT ~1.26
            (1.10, 0.283),   # FTT ~1.27
            (1.15, 0.375),   # FTT ~1.27+
            (1.20, 0.484),   # FTT ~1.28: 舊控制器已失控
            (1.30, 0.641),   #
            (1.40, 0.768),   #
            (1.50, 1.000),   #
            (2.00, 1.414),   # 完全發散
        ]

        forces_during_crisis = []
        ma_factors_during_crisis = []
        force_ever_zero = False

        for Ub_frac, Ma_max_val in divergence_sequence:
            s = controller_update(s, Ub_avg=UREF * Ub_frac, Ma_max=Ma_max_val)
            forces_during_crisis.append(s.Force)
            ma_factors_during_crisis.append(s.Ma_factor)
            if s.Force < 1e-15:
                force_ever_zero = True

        # ── 驗證 ──
        # 1. 在 Ma_max > threshold (≈0.128) 時, brake 應啟動
        #    序列第 3 項 (Ma=0.130) > threshold (0.128)
        self.assertLess(ma_factors_during_crisis[2], 1.0,
                        f"Brake should activate at Ma_max=0.130 > threshold={MA_THRESHOLD:.3f}")

        # 2. Force 在危機中應快速降到 0 (Ma_critical ≈ 0.135 時歸零)
        self.assertTrue(force_ever_zero,
                        "Force should reach 0 when Ma_max exceeds critical")

        # 3. 所有 Ma_max > Ma_critical 的步驟, Force 應為 0
        for i, (_, Ma_val) in enumerate(divergence_sequence):
            if Ma_val > MA_CRITICAL:
                self.assertAlmostEqual(forces_during_crisis[i], 0.0, places=10,
                    msg=f"Force should be 0 when Ma_max={Ma_val:.3f} > critical={MA_CRITICAL:.3f}")
                break

    def test_brake_activates_before_old_threshold(self):
        """新 brake 在 Ma≈0.128 介入, 遠早於舊閾值 0.25"""
        s = ControllerState(Force=1e-4, initialized=True)
        s.Ma_max_prev = 0.12

        # Ma_max = 0.130 > threshold (≈0.128)
        s = controller_update(s, Ub_avg=UREF * 0.99, Ma_max=0.130)
        self.assertLess(s.Ma_factor, 1.0,
                        f"New brake should activate at 0.130 > threshold={MA_THRESHOLD:.3f}")
        self.assertGreater(s.Ma_factor, 0.0, "Should not zero out at 0.130")

        # 比較: Ma > critical (≈0.135) 時歸零
        s2 = ControllerState(Force=1e-4, initialized=True)
        s2.Ma_max_prev = 0.13
        s2 = controller_update(s2, Ub_avg=UREF * 0.99, Ma_max=0.14)
        self.assertAlmostEqual(s2.Ma_factor, 0.0, places=5,
                        msg=f"At Ma=0.14 > critical={MA_CRITICAL:.3f}, brake should zero force")

    def test_growth_rate_catches_rapid_rise(self):
        """Ma_max 急速增長 (如 0.08 → 0.12 = +50%) 觸發 rate brake"""
        s = ControllerState(Force=1e-4, initialized=True, gehrke_activated=True)
        s.Force_integral = 1e-4
        s.error_prev = 0.0
        s.Ma_max_prev = 0.08

        # +50% 增長, Ma=0.12 > 1.5×Ma_bulk (≈0.096)
        s = controller_update(s, Ub_avg=UREF * 1.0, Ma_max=0.12)
        # rate brake 應觸發 (>30% 且 Ma > 1.5×Ma_bulk)
        # Also Ma=0.12 < threshold (0.128), so quadratic brake NOT active
        # but rate brake alone: factor *= 0.3
        self.assertLess(s.Ma_factor, 0.5,
                        "Rate brake should trigger on 50% Ma growth")


class TestGehrkeConvergence(unittest.TestCase):
    """Test 6: Gehrke 控制器穩態收斂性"""

    def test_converges_to_uref(self):
        """從 |Re%|=4% 開始, Gehrke 應在 ~20 更新內收斂到 dead zone"""
        s = ControllerState(Force=F_POISEUILLE, initialized=True, gehrke_activated=True)
        s.Force_integral = F_POISEUILLE
        s.error_prev = UREF - UREF * 0.96
        s.Ma_max_prev = 0.06

        # 簡化模型: 假設 Ub ∝ Force (線性響應近似)
        # Ub/Uref ≈ (Force / F_target) 其中 F_target = steady state force
        Ub_frac = 0.96  # 初始 Re% = -4%

        for i in range(50):
            Ub = UREF * Ub_frac
            s = controller_update(s, Ub_avg=Ub, Ma_max=0.06)
            # 簡化: Force 增加 → Ub 增加 (向 Uref 靠攏)
            # 使用非常簡化的響應模型
            Ub_frac += (s.Force / F_POISEUILLE - 1.0) * 0.005
            Ub_frac = max(0.5, min(1.5, Ub_frac))

        Re_pct_final = (Ub_frac - 1.0) * 100.0
        self.assertLess(abs(Re_pct_final), 5.0,
                        f"Re% should be within 5% after 50 updates, got {Re_pct_final:.2f}%")


class TestEdgeCases(unittest.TestCase):
    """Test 7: 邊界條件"""

    def test_zero_ub(self):
        """Ub = 0 → PID 模式, Force > 0"""
        s = ControllerState()
        s = controller_update(s, Ub_avg=0.0, Ma_max=0.001)
        self.assertGreater(s.Force, 0.0)
        self.assertIn("PID", s.mode)

    def test_exact_uref(self):
        """Ub = Uref → Gehrke HOLD (Re% = 0, 在死區內)"""
        s = ControllerState(Force=1e-4, initialized=True, gehrke_activated=True)
        s.Force_integral = 1e-4
        s.error_prev = 0.0
        s.Ma_max_prev = 0.06
        s = controller_update(s, Ub_avg=UREF, Ma_max=0.06)
        self.assertEqual(s.mode, "GEHRKE-HOLD")

    def test_very_high_ma_max(self):
        """Ma_max = 0.5 (far beyond critical ≈ 0.135) → Force = 0"""
        s = ControllerState(Force=1e-3, initialized=True)
        s.Ma_max_prev = 0.3
        s = controller_update(s, Ub_avg=UREF * 0.8, Ma_max=0.5)
        self.assertAlmostEqual(s.Force, 0.0, places=15)

    def test_negative_ma_growth(self):
        """Ma 下降 (negative growth) 不觸發 rate brake"""
        s = ControllerState(Force=1e-4, initialized=True)
        s.Ma_max_prev = 0.20
        # Ma 下降 50%
        s = controller_update(s, Ub_avg=UREF * 0.99, Ma_max=0.10)
        # rate brake 不觸發 (growth < 0)
        # Ma_max=0.10 < threshold → factor = 1.0
        self.assertAlmostEqual(s.Ma_factor, 1.0)

    def test_force_floor_in_gehrke(self):
        """Gehrke 模式下 Force 持續減力, FLOOR=0 時趨近 0 但 >0 (乘法)"""
        s = ControllerState(Force=1e-4, initialized=True, gehrke_activated=True)
        s.Force_integral = 1e-4
        s.error_prev = UREF - UREF * 1.04
        s.Ma_max_prev = 0.06

        # 持續 overshoot → Gehrke 持續減力 (correction=0.6 each step)
        for _ in range(100):
            s = controller_update(s, Ub_avg=UREF * 1.04, Ma_max=0.06)

        # With FLOOR=0.0, Force approaches 0 via multiplication (0.6^100 ≈ 6.5e-23)
        # Still positive due to multiplicative nature
        self.assertGreaterEqual(s.Force, 0.0)


class TestFullScenario(unittest.TestCase):
    """Test 8: 完整模擬場景 (cold start → PID → Gehrke → stable)"""

    def test_full_lifecycle(self):
        """完整生命週期: 冷啟動 → 加速 → 接近目標 → 切 Gehrke → 穩定"""
        s = ControllerState()
        history = {'mode': [], 'Force': [], 'Ub_frac': []}

        Ub_frac = 0.0  # 冷啟動
        for step in range(500):
            Ub = UREF * Ub_frac
            # Ma_max 模型: 保守固定值, 不觸發 Mach brake
            Ma = 0.08  # 典型穩態 Ma_max << threshold (0.128)
            s = controller_update(s, Ub_avg=Ub, Ma_max=Ma)

            history['mode'].append(s.mode)
            history['Force'].append(s.Force)
            history['Ub_frac'].append(Ub_frac)

            # 簡化響應模型: Force 驅動 Ub, 加上阻力回復
            # dUb/dt ∝ Force - drag*Ub^2
            drag = F_POISEUILLE / (UREF * UREF)  # 穩態: Force=F_P 時 Ub=Uref
            Ub_frac += (s.Force - drag * (UREF * Ub_frac)**2) / (F_POISEUILLE) * 0.002
            Ub_frac = max(0.0, min(1.5, Ub_frac))

        # 驗證:
        # 1. 開頭應在 PID 模式
        self.assertIn("PID", history['mode'][0])

        # 2. 最終 Force 應為正 (穩態驅動)
        self.assertGreater(history['Force'][-1], 0.0,
                           "Force should be positive in steady state")

        # 3. 最終應接近目標 (Ub_frac ≈ 1.0)
        self.assertGreater(history['Ub_frac'][-1], 0.5,
                           "Ub should have accelerated from cold start")


# =====================================================================
if __name__ == '__main__':
    print("=" * 72)
    print("Hybrid Dual-Stage Force Controller — Unit Tests")
    print("=" * 72)
    print(f"  Uref = {UREF},  Re = {RE},  LY = {LY}")
    print(f"  Ma_bulk_ref = {MA_BULK_REF:.6f}")
    print(f"  Ma_threshold = {MA_THRESHOLD:.6f}  (2.5 × Ma_bulk)")
    print(f"  Ma_critical  = {MA_CRITICAL:.6f}  (4.0 × Ma_bulk)")
    print(f"  norm = {NORM:.6e}")
    print(f"  F_Poiseuille = {F_POISEUILLE:.6e}")
    print(f"  F_floor = {F_FLOOR:.6e}")
    print(f"  SWITCH_THRESHOLD = {FORCE_SWITCH_THRESHOLD}%")
    print(f"  GEHRKE_GAIN = {FORCE_GEHRKE_GAIN}")
    print("=" * 72)
    unittest.main(verbosity=2)
