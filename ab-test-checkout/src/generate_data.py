"""
Генерация синтетических данных A/B-теста нового экрана оплаты.

Сценарий: интернет-магазин тестирует редизайн экрана оплаты
(одна страница вместо трёх шагов, сохранённые карты, Apple/Google Pay).
Пользователи, дошедшие до экрана оплаты, случайно делятся 50/50:
  A — контроль (старый экран), B — тест (новый экран).

Чтобы данные были похожи на реальные, в них специально заложены артефакты:
  * дубли событий (повторная отправка лога);
  * пользователи, попавшие в обе группы (баг сплитования);
  * технический тестовый трафик (QA-аккаунты).

Запуск:  python src/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_USERS = 52_000
START = pd.Timestamp("2026-09-01")
DAYS = 14

# Базовая конверсия в покупку по сегментам (контроль)
BASE_CR = {
    ("ios", "new"): 0.22, ("ios", "returning"): 0.33,
    ("android", "new"): 0.18, ("android", "returning"): 0.28,
    ("web", "new"): 0.20, ("web", "returning"): 0.31,
}
# Абсолютный прирост конверсии в тесте: новый экран сильнее помогает на мобильных
UPLIFT = {"ios": 0.022, "android": 0.024, "web": 0.004}

OUT_DIR = Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    rng = np.random.default_rng(SEED)

    users = pd.DataFrame({"user_id": np.arange(100_001, 100_001 + N_USERS)})
    users["group"] = rng.choice(["A", "B"], size=N_USERS)
    users["platform"] = rng.choice(["ios", "android", "web"], size=N_USERS, p=[0.38, 0.34, 0.28])
    users["user_type"] = rng.choice(["new", "returning"], size=N_USERS, p=[0.45, 0.55])

    # Дата первого захода на экран оплаты: в выходные трафик выше
    day_weights = np.array([1.25 if (START + pd.Timedelta(days=d)).dayofweek >= 5 else 1.0 for d in range(DAYS)])
    day = rng.choice(DAYS, size=N_USERS, p=day_weights / day_weights.sum())
    seconds = rng.integers(0, 24 * 3600, size=N_USERS)
    users["event_time"] = START + pd.to_timedelta(day, unit="D") + pd.to_timedelta(seconds, unit="s")

    # Конверсия
    p = users.apply(lambda r: BASE_CR[(r.platform, r.user_type)], axis=1).to_numpy()
    p = p + np.where(users["group"] == "B", users["platform"].map(UPLIFT).to_numpy(), 0.0)
    # Небольшой эффект новизны: в первые 3 дня тест чуть сильнее
    p = p + np.where((users["group"] == "B") & (day < 3), 0.008, 0.0)
    users["converted"] = (rng.random(N_USERS) < p).astype(int)

    # Сумма заказа (руб.): логнормальное распределение с тяжёлым хвостом.
    # В тесте средний чек почти не меняется (чуть больше мелких «импульсных» покупок)
    mu = np.where(users["user_type"] == "returning", 8.15, 7.95)
    mu = mu - np.where(users["group"] == "B", 0.01, 0.0)
    revenue = np.round(rng.lognormal(mean=mu, sigma=0.75), -1)
    users["revenue"] = np.where(users["converted"] == 1, revenue, 0.0)

    # ---------- «Грязь» в данных ----------
    # 1) ~0.6% пользователей попали в обе группы
    cross = users.sample(frac=0.006, random_state=SEED).copy()
    cross["group"] = np.where(cross["group"] == "A", "B", "A")
    cross["event_time"] = cross["event_time"] + pd.to_timedelta(rng.integers(60, 86_400, len(cross)), unit="s")
    # 2) ~1% дублей строк
    dups = users.sample(frac=0.01, random_state=SEED + 1)
    # 3) QA-трафик: 150 тестовых аккаунтов, только в группе B, все «покупают»
    qa = pd.DataFrame({
        "user_id": np.arange(900_001, 900_151),
        "group": "B",
        "platform": rng.choice(["ios", "android", "web"], 150),
        "user_type": "returning",
        "event_time": START + pd.to_timedelta(rng.integers(0, DAYS * 86_400, 150), unit="s"),
        "converted": 1,
        "revenue": 1.0,
    })

    df = pd.concat([users, cross, dups, qa], ignore_index=True)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["event_time"] = df["event_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df[["user_id", "group", "event_time", "platform", "user_type", "converted", "revenue"]]

    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "checkout_ab_test.csv", index=False)
    print(f"Saved {len(df):,} rows -> {OUT_DIR / 'checkout_ab_test.csv'}")


if __name__ == "__main__":
    main()
