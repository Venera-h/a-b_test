"""Функции для анализа A/B-тестов: SRM, мощность, тесты, бутстрап."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


# ---------------------------------------------------------------- дизайн теста
def sample_size_proportion(p_base: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Размер выборки на одну группу для теста долей (двусторонний)."""
    effect = proportion_effectsize(p_base + mde_abs, p_base)
    n = NormalIndPower().solve_power(effect_size=effect, alpha=alpha, power=power, ratio=1.0)
    return int(np.ceil(n))


def mde_proportion(p_base: float, n_per_group: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Минимальный детектируемый абсолютный эффект при заданном размере групп."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return z * np.sqrt(2 * p_base * (1 - p_base) / n_per_group)


# ---------------------------------------------------------------- SRM
def srm_check(n_a: int, n_b: int, expected_share_a: float = 0.5) -> float:
    """Sample Ratio Mismatch: хи-квадрат тест на соответствие сплиту. Возвращает p-value."""
    total = n_a + n_b
    expected = [total * expected_share_a, total * (1 - expected_share_a)]
    return stats.chisquare([n_a, n_b], f_exp=expected).pvalue


# ---------------------------------------------------------------- тесты
@dataclass
class TestResult:
    metric: str
    control: float
    treatment: float
    abs_diff: float
    rel_diff: float
    ci_low: float
    ci_high: float
    p_value: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def ztest_proportions(conv_a: np.ndarray, conv_b: np.ndarray, alpha: float = 0.05,
                      metric: str = "conversion") -> TestResult:
    """Z-тест для разницы долей + доверительный интервал разницы (Вальд)."""
    n_a, n_b = len(conv_a), len(conv_b)
    p_a, p_b = conv_a.mean(), conv_b.mean()
    p_pool = (conv_a.sum() + conv_b.sum()) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pool
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    q = stats.norm.ppf(1 - alpha / 2)
    diff = p_b - p_a
    return TestResult(metric, p_a, p_b, diff, diff / p_a, diff - q * se, diff + q * se, p_value)


def welch_ttest(x_a: np.ndarray, x_b: np.ndarray, alpha: float = 0.05, metric: str = "mean") -> TestResult:
    """t-тест Уэлча для средних + ДИ разницы."""
    m_a, m_b = x_a.mean(), x_b.mean()
    res = stats.ttest_ind(x_b, x_a, equal_var=False)
    v_a, v_b = x_a.var(ddof=1) / len(x_a), x_b.var(ddof=1) / len(x_b)
    se = np.sqrt(v_a + v_b)
    dof = (v_a + v_b) ** 2 / (v_a ** 2 / (len(x_a) - 1) + v_b ** 2 / (len(x_b) - 1))
    q = stats.t.ppf(1 - alpha / 2, dof)
    diff = m_b - m_a
    return TestResult(metric, m_a, m_b, diff, diff / m_a, diff - q * se, diff + q * se, res.pvalue)


# ---------------------------------------------------------------- бутстрап
def bootstrap_diff(x_a: np.ndarray, x_b: np.ndarray, stat=np.mean, n_boot: int = 5_000,
                   alpha: float = 0.05, seed: int = 0) -> dict:
    """Бутстрап-распределение разницы статистики (B − A), перцентильный ДИ и p-value."""
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = x_a[rng.integers(0, len(x_a), len(x_a))]
        sb = x_b[rng.integers(0, len(x_b), len(x_b))]
        diffs[i] = stat(sb) - stat(sa)
    ci = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    # p-value: насколько ноль «далеко» в распределении разницы (двусторонний)
    p_value = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diffs": diffs, "observed": stat(x_b) - stat(x_a),
            "ci_low": ci[0], "ci_high": ci[1], "p_value": max(p_value, 1 / n_boot)}


# ---------------------------------------------------------------- A/A проверка метода
def aa_false_positive_rate(values: np.ndarray, test_fn, n_sim: int = 1_000, alpha: float = 0.05,
                           seed: int = 0) -> float:
    """Доля ложных срабатываний теста на случайных A/A-разбиениях одной выборки."""
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sim):
        mask = rng.random(len(values)) < 0.5
        if test_fn(values[~mask], values[mask]).p_value < alpha:
            hits += 1
    return hits / n_sim
