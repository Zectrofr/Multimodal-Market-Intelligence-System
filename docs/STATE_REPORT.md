# MMIS (Fincorp) — State Report

Read-only audit. No source file, database, or artifact was modified. The pipeline was not run.

---

## 0. Meta

| Field | Value |
|---|---|
| Report date | 2026-08-12 |
| Python | 3.10.11 (`C:\Users\Khemraj\AppData\Local\Programs\Python\Python310\python.exe`, system install, **no venv**) |
| OS | Microsoft Windows 11 Home Single Language, NT 10.0.26200 |
| Shell | PowerShell 5.1 |
| Git branch | `main` |
| HEAD | `c6497e9` — "feat(layer-6): live demo — on-demand inference, Grad-CAM, FastAPI + Streamlit" |
| Working tree | Clean of modifications |
| Uncommitted (modified tracked) files | **0** |
| Untracked files | **2** (`scripts/audit_state.py`, `docs/STATE_REPORT.md` — both created by this audit) |
| Remote configured | Yes — one remote, name `origin` (URL not retrieved) |
| Upstream sync | `origin/main`, 0 ahead / 0 behind |
| Total commits | 16 |
| Tracked files in repo | 17 |
| Repo state | Not mid-rebase, not mid-merge, not detached HEAD |

Last commit to the repository was 2026-06-15; the audit runs ~2 months later. No source file has changed in that window.

### Database tamper check

| | Length (bytes) | LastWriteTime |
|---|---|---|
| Before audit | 55,296,000 | 14-06-2026 20:42:23 |
| After audit | 55,296,000 | 14-06-2026 20:42:23 |

**PASS** — `data\mmis.db` is byte-identical and its mtime is unchanged. All access used `sqlite3.connect("file:data/mmis.db?mode=ro", uri=True)`.

---

## 1. Repo inventory

### Directory tree (depth 3; excludes `.git`, `__pycache__`, `mlruns` internals, image dirs)

```
Fincorp\
  .env  .gitignore  README.md  requirements.txt  mlflow.db
  evaluation.py  fusion.py  regime.py  ingest.py  live_inference.py
  inference.py  vision.py  sentiment.py  gradcam.py
  .claude\      settings.local.json
  .vscode\      settings.json
  api\          __init__.py  main.py  __pycache__\
  dashboard\    app.py  __pycache__\
  scripts\      robustness_sweep.py  audit_state.py(new)
  data\         mmis.db
                images\charts\        -> 9,310 PNG, 47.53 MB
  models\       10 files (see §6)
  results\      39 files, 2.31 MB (see §6)
  mlruns\       1\  -> 112 files, 67.23 MB (artifacts only; see §6)
  docs\         STATE_REPORT.md(new)
```

No `venv/.venv/env`, no `node_modules`, no `tests\`, no `.github\`.

### Python files (line counts are true totals via `wc -l`)

| Path | Lines | Last modified |
|---|---|---|
| `regime.py` | 844 | 2026-06-14 20:41 |
| `evaluation.py` | 793 | 2026-06-15 15:56 |
| `fusion.py` | 569 | 2026-06-15 00:12 |
| `ingest.py` | 316 | 2026-06-14 21:17 |
| `live_inference.py` | 256 | 2026-06-15 21:50 |
| `inference.py` | 252 | 2026-06-15 19:51 |
| `vision.py` | 184 | 2026-06-11 14:25 |
| `sentiment.py` | 163 | 2026-06-10 04:00 |
| `dashboard\app.py` | 109 | 2026-06-15 22:31 |
| `gradcam.py` | 107 | 2026-06-15 22:07 |
| `api\main.py` | 99 | 2026-06-15 22:29 |
| `scripts\robustness_sweep.py` | 63 | 2026-06-15 20:12 |
| `api\__init__.py` | 0 | 2026-06-15 22:29 |
| **Total (pre-existing)** | **3,755** | |
| `scripts\audit_state.py` (created by this audit) | 215 | 2026-08-12 |

### Sizes

| Item | Size |
|---|---|
| Repo total incl. `.git` | 301.40 MB |
| `.git` | 121.80 MB |
| Repo excl. `.git` | 179.60 MB |
| **`data\mmis.db`** | **55,296,000 bytes (52.73 MiB)** |
| `data\images\charts\` | 47.53 MB (9,310 PNG) |
| `mlruns\` | 67.23 MB |
| `models\` | 8.43 MB |
| `results\` | 2.31 MB |
| `mlflow.db` | 1,155,072 bytes |

`.git` is 121.8 MB against 17 tracked text files (~180 KB) — history contains large binaries added before the gitignore cleanups in `2bbf173` / `7a9097c`.

---

## 2. Environment and dependencies

### Installed versions (all present; none missing)

| Package | Installed | Package | Installed |
|---|---|---|---|
| torch | 2.12.0+cpu | mlflow | 3.13.0 |
| torchvision | 0.27.0 | fastapi | 0.136.3 |
| transformers | 5.10.2 | uvicorn | 0.49.0 |
| scikit-learn | 1.7.2 | streamlit | 1.58.0 |
| hmmlearn | 0.3.3 | flask | 3.1.3 |
| pandas | 2.3.3 | scipy | 1.15.3 |
| numpy | 2.2.6 | matplotlib | 3.10.9 |
| SQLAlchemy | 2.0.50 | newsapi-python | 0.2.7 |
| mplfinance | 0.12.10b0 | ta | 0.11.0 |
| yfinance | 1.4.1 | starlette | 1.3.0 |

Other web/API-related installed: `flask-cors 6.0.5`, `waitress 3.0.2`, `httpx 0.28.1`, `requests 2.34.2`, `aiohttp 3.14.1`, `websockets 16.0`, `python-multipart 0.0.32`, `psycopg2-binary 2.9.12`, `alembic 1.18.4`. `gunicorn` NOT installed.

152 packages total in a **system-wide** interpreter — unrelated heavy libraries (`pytorch-forecasting`, `lightning`, `huey`, `peewee`, `opencv-python`) share the namespace, so `pip freeze` cannot regenerate a clean environment.

### CUDA

```
torch.__version__      = 2.12.0+cpu
torch.cuda.is_available() = False
torch.cuda.device_count() = 0
```

**CPU-only build.** No model was loaded. All timings in `results\*.log` are CPU numbers.

### Manifests

`requirements.txt` exists (29 lines). `pyproject.toml`, `environment.yml`, `setup.py`, `setup.cfg`, `Pipfile`, `Makefile`, `Dockerfile`, `CLAUDE.md` — **none exist**.

Agreement between pins and installed:

| Package | Pinned | Installed | Verdict |
|---|---|---|---|
| `python-dotenv` (`requirements.txt:28`) | 1.0.1 | 1.2.2 | **MISMATCH** |
| `torch` (`requirements.txt:14`) | 2.12.0 | 2.12.0+cpu | Version matches; pin does not encode the CPU/CUDA build variant |
| all other 24 pins | — | — | match exactly |

**`tqdm` is imported at `vision.py:20` but absent from `requirements.txt`.** A clean `pip install -r requirements.txt` leaves `vision.py` un-runnable. `opencv-python` is installed but imported by no project file.

### Config files (key names only — no values read or printed)

| File | Present | Keys |
|---|---|---|
| `.env` (45 B) | Yes | **`NEWS_API_KEY`: present, non-empty, redacted** — the only key |
| `config.py` / `settings.yaml` | No | — |
| `.gitignore` (16 lines) | Yes | ignores `.env`, `data/mmis.db`, `data/images/`, `__pycache__/`, `mlruns/`, `mlflow.db`, `models/*.pt`, `models/*.pkl`, `results/` |
| `.vscode\settings.json` | Yes | `python-envs.defaultEnvManager` = system interpreter |
| `.claude\settings.local.json` | Yes | permissions allowlist only; no secrets, no hooks |

Env vars read from the process environment (not `.env`): `MMIS_SEED` default `"42"` (`fusion.py:43`), `MMIS_API_URL` default `http://127.0.0.1:8000` (`dashboard\app.py:16`).

Note: the CONTEXT brief refers to `NEWSAPI_KEY`; the actual key name is `NEWS_API_KEY`, matching `sentiment.py:31`.

---

## 3. Database reality

`data\mmis.db` — SQLite 3.40.1, page_size 4096, page_count 13,500, `journal_mode=delete`, `user_version=0`.

### 3.1 Tables, columns, row counts

**Four tables.** No views, no triggers.

| Table | Rows | Columns |
|---|---|---|
| `market_data` | **9,412** | 44 |
| `visual_features` | **9,292** | 4 |
| `sentiment_data` | **124** | 7 |
| `chart_images` | **0** | 4 |

**`market_data`** (44 cols; declared types in parentheses)
`date`(TIMESTAMP, PK1), `ticker`(TEXT, PK2), `open`(REAL), `high`(REAL), `low`(REAL), `close`(REAL), `volume`(REAL), `rsi_14`(REAL), `macd`(REAL), `macd_signal`(REAL), `macd_diff`(REAL), `ema_20`(REAL), `ema_50`(REAL), `bb_upper`(REAL), `bb_lower`(REAL), `bb_mid`(REAL), `bb_bandwidth`(REAL), `bb_position`(REAL), `atr_14`(REAL), `returns_1d`(REAL), `returns_5d`(REAL), `returns_10d`(REAL), `returns_20d`(REAL), `volume_ma_20`(REAL), `volume_ratio`(REAL), `volume_change_1d`(REAL), `body_size`(REAL), `upper_shadow`(REAL), `lower_shadow`(REAL), `high_low_range`(REAL), `day_of_week`(INTEGER), `month`(INTEGER), `day_of_year`(INTEGER), `day_of_year_sin`(REAL), `day_of_year_cos`(REAL), `month_sin`(REAL), `month_cos`(REAL), `time_idx`(INTEGER), `group_id`(TEXT), `target`(INTEGER), `split`(TEXT DEFAULT 'train'), `created_at`(TIMESTAMP DEFAULT CURRENT_TIMESTAMP), `regime`(TEXT), `regime_id`(INTEGER)

**`sentiment_data`** (7 cols): `date`(TEXT), `ticker`(TEXT), `sentiment_neg`(FLOAT), `sentiment_neu`(FLOAT), `sentiment_pos`(FLOAT), `sentiment_score`(FLOAT), `headline_count`(BIGINT)

**`visual_features`** (4 cols): `date`(TEXT), `ticker`(TEXT), `image_path`(TEXT), `feature_vector`(**declared TEXT — actually stores BLOB**)

**`chart_images`** (4 cols): `date`(TIMESTAMP, PK1), `ticker`(TEXT, PK2), `image_path`(TEXT), `window_size`(INTEGER)

### 3.2 Indexes, primary keys, uniqueness, foreign keys

| Table | PK | Indexes | FK |
|---|---|---|---|
| `market_data` | `(date, ticker)` | `sqlite_autoindex_market_data_1` (UNIQUE, cols date+ticker) | none |
| `chart_images` | `(date, ticker)` | `sqlite_autoindex_chart_images_1` (UNIQUE, cols date+ticker) | `(date,ticker)` → `market_data(date,ticker)` |
| `sentiment_data` | **NONE** | **NONE** | none |
| `visual_features` | **NONE** | **NONE** | none |

`sentiment_data` and `visual_features` were declared with `PRIMARY KEY (date, ticker)` in code (`sentiment.py:105-116`, `vision.py:128-136`), but those DDLs are destroyed by `to_sql(..., if_exists="replace")` at `sentiment.py:152` and `vision.py:173`, which drops and recreates the table with pandas-inferred types. **No uniqueness constraint protects either table**, and every join against them is a full scan.

### 3.3 Per-ticker coverage

`market_data` — 6 distinct tickers:

| ticker | rows | min date | max date |
|---|---|---|---|
| AAPL | 1,568 | 2020-03-13 | 2026-06-09 |
| AMZN | 1,569 | 2020-03-13 | 2026-06-10 |
| GOOGL | 1,568 | 2020-03-13 | 2026-06-09 |
| MSFT | 1,569 | 2020-03-13 | 2026-06-10 |
| SPY | 1,569 | 2020-03-13 | 2026-06-10 |
| TSLA | 1,569 | 2020-03-13 | 2026-06-10 |

Series begins 2020-03-13, not the configured `START_DATE = "2020-01-01"` (`ingest.py:25`) — `df.dropna()` at `ingest.py:143` removes the 49-row `ema_50` warm-up per ticker.

`visual_features`: 1,548–1,549 rows per ticker, 2020-04-13 → 2026-06-09/10.
`sentiment_data`: 20–21 rows per ticker, **2026-05-12 → 2026-06-10 only**.

`market_data` rows by year: 2020: 1,224 · 2021: 1,512 · 2022: 1,506 · 2023: 1,500 · 2024: 1,512 · 2025: 1,500 · 2026: 658.

### 3.4 Date column storage format — CRITICAL

All three populated date columns are stored as **TEXT**, uniform `LENGTH = 26`, format `'YYYY-MM-DD HH:MM:SS.ffffff'`.

| Column | Declared type | `typeof()` histogram | Length histogram |
|---|---|---|---|
| `market_data.date` | TIMESTAMP | `text` × 9,412 | 26 × 9,412 |
| `sentiment_data.date` | TEXT | `text` × 124 | 26 × 124 |
| `visual_features.date` | TEXT | `text` × 9,292 | 26 × 9,292 |

Three literal example values, verbatim (Python `repr`):

```
'2020-03-13 00:00:00.000000'
'2020-04-13 00:00:00.000000'
'2026-06-10 00:00:00.000000'
```

There is **no format variation** — every row in every table carries the microsecond suffix. Any code comparing against `strftime("%Y-%m-%d")` (`'2020-03-13'`) matches **zero rows** on `=`. Ordering by this TEXT column happens to be chronological only because the format is zero-padded and fixed-width.

`market_data.date` is declared `TIMESTAMP` but SQLite stores TEXT; `visual_features.feature_vector` is declared `TEXT` but every value is `typeof() = 'blob'`. **Declared types are unreliable in this database.**

### 3.5 NULL counts

**Every column of every table has 0 NULLs (0.00%).** `chart_images` is empty so the question does not arise there.

| Table | Columns checked | NULLs found |
|---|---|---|
| `market_data` | 44 | 0 in all 44 |
| `sentiment_data` | 7 | 0 in all 7 |
| `visual_features` | 4 | 0 in all 4 |
| `chart_images` | 4 | table empty (0 rows) |

**Rows with a NULL `regime`: 0 (0.00% of 9,412).** The regime column is fully populated:

| regime | rows |
|---|---|
| mean_reverting | 3,834 |
| trending | 3,683 |
| high_vol | 1,895 |

`regime_id` is also 0% NULL, distributed 0: 3,754 / 1: 3,149 / 2: 2,509 — **which does not reconcile with the `regime` marginals above.** See §5 "Unlisted defects found", U1.

### 3.6 Sentiment coverage

**Method.** Sentiment lives in a separate table, so coverage is measured as `sentiment_data` rows joinable to `market_data` on exact `(date, ticker)`. "Default/placeholder" was determined as the **modal tuple** across `(sentiment_neg, sentiment_neu, sentiment_pos, sentiment_score)`, then cross-checked against `headline_count`.

| Quantity | Count | % of `market_data` (9,412) |
|---|---|---|
| `sentiment_data` rows total | 124 | **1.3175%** |
| Rows joining `market_data` on exact `(date,ticker)` | 124 | 1.3175% (100% of sentiment rows join) |
| **Modal / placeholder rows** | **24** | 0.2550% |
| **Genuinely non-default rows** | **100** | **1.0625%** |

- Modal tuple: **`(0.0, 1.0, 0.0, 0.0)`**, frequency **24 / 124 = 19.35%** of the sentiment table.
- Distinct tuples: 101. Distinct values per column: 101 each.
- The 24 placeholder rows correspond exactly to `headline_count = 0`; the `headline_count` histogram is `0: 24, 8: 4, 9: 14, 10: 82`. The placeholder is therefore identifiable, but only via `headline_count` — the three probability columns are indistinguishable from a genuine perfectly-neutral reading.
- Placeholders are spread evenly: exactly 4 per ticker.
- **All 124 rows fall in calendar year 2026**, spanning 2026-05-12 → 2026-06-10 — a ~30-day window against a market table spanning 2020-03-13 → 2026-06-10. This is the NewsAPI free-tier window made visible in the data.

### 3.7 Vision features

| Quantity | Value |
|---|---|
| Rows with non-NULL `feature_vector` | **9,292 / 9,292 (100% of the table)** |
| Coverage against `market_data` | 9,292 / 9,412 = **98.72%** |
| `market_data` rows with no visual match | **120** |
| BLOB byte length | **5,120 bytes — uniform across all 9,292 rows** (single-value histogram) |
| Implied dtype/dim | 5,120 / 4 = **1,280 float32** |
| `typeof()` | `blob` × 9,292 (despite declared TEXT) |

BLOB contents were not printed. The 120 unmatched rows are the first 20 bars per ticker (`vision.py:152` starts the loop at index 20).

Nothing in the database records the dtype, shape, or endianness of the BLOB — it is a bare `numpy.ndarray.tobytes()` dump (`vision.py:165`). Readers must hardcode `np.frombuffer(..., dtype=np.float32)`.

### 3.8 Market feature dimensionality (input to D6)

`market_data` has 44 columns. Excluding 9 non-feature columns (`date`, `ticker`, `group_id`, `target`, `split`, `created_at`, `regime`, `regime_id`, `time_idx`) leaves **35 candidate numeric feature columns**:

```
open, high, low, close, volume, rsi_14, macd, macd_signal, macd_diff,
ema_20, ema_50, bb_upper, bb_lower, bb_mid, bb_bandwidth, bb_position,
atr_14, returns_1d, returns_5d, returns_10d, returns_20d, volume_ma_20,
volume_ratio, volume_change_1d, body_size, upper_shadow, lower_shadow,
high_low_range, day_of_week, month, day_of_year, day_of_year_sin,
day_of_year_cos, month_sin, month_cos
```

`fusion.py:266-275` (`MARKET_COLS`) selects **31** of these, omitting `volume_ma_20`, `day_of_week`, `month`, `day_of_year`. No column in `MARKET_COLS` is missing from the DB.

### 3.9 Anomalies

- **`chart_images` is empty (0 rows)** despite being created at `ingest.py:226-234` with a foreign key to `market_data`. No module in the repo ever inserts into it. Dead table.
- **No duplicate `(date, ticker)` rows** in any of the three populated tables.
- `market_data.split` is populated: `train` 7,528 (2020-03-13 → 2025-03-11), `validation` 1,884 (2025-03-11 → 2026-06-10). **The boundary date 2025-03-11 appears in both splits.** This column is written by `ingest.py:243-245` (per-ticker 80/20) and is *not* the split `fusion.py` actually uses.
- `market_data.target` distribution: 0 (Down) 3,153 · 1 (Flat) 2,598 · 2 (Up) 3,661.
- `created_at` shows six distinct timestamps on 2026-06-11 (09:43–10:02), one per ticker ingest — confirming a single ingest run.
- `time_idx` restarts at 0 for every ticker (range 0→1,567/1,568), so it is not a global ordering key.

---

## 4. Module map

Pipeline order as documented: ingest → sentiment → vision → regime → fusion → inference → evaluation.

### 4.1 `ingest.py` (316 lines)

Downloads adjusted OHLCV for 6 hardcoded tickers from yfinance, computes 33 derived columns including a 3-class next-day target, renders one 20-day candlestick PNG per row, and bulk-appends to `market_data`.

**Functions** (no classes): `fetch_ohlcv(ticker, start=START_DATE, end=END_DATE) -> pd.DataFrame` :36 · `add_indicators(df) -> pd.DataFrame` :64 · `generate_chart(df, ticker, date_idx, window=20, size=(224,224)) -> Path` :151 · `init_database(engine)` :195 · `save_to_db(df, engine, ticker)` :240 · `run_pipeline(tickers=TICKERS, generate_charts=True)` :255

**Entry point** `:309`. argparse: `--tickers` (nargs="+", default `TICKERS`) :311 · `--no-charts` (store_true) :312. No `--start`/`--end`/`--db`.

**DB reads:** none — pure producer.
**DB writes:** `CREATE TABLE IF NOT EXISTS market_data` (42 cols, PK `(date,ticker)`) :199-222 · `CREATE TABLE IF NOT EXISTS chart_images` :225-234 (never populated) · `df.to_sql("market_data", if_exists="append", index=False)` :251 — writes all 41 DataFrame columns; `created_at` comes from the DB default.

**Constants:** `TICKERS = ["AAPL","GOOGL","MSFT","TSLA","AMZN","SPY"]` :24 · `START_DATE = "2020-01-01"` :25 · `END_DATE = datetime.today().strftime("%Y-%m-%d")` :26 (**evaluated at import — non-reproducible**) · `DB_PATH = "sqlite:///data/mmis.db"` :27 · `CHART_DIR = Path("data/images/charts")` :28 · `FLAT_THRESHOLD = 0.005` :29. In-function: RSI 14 :73 · MACD 26/12/9 :76 · EMA 20/50 :82-83 · Bollinger 20/2 :87 · ATR 14 :95 · return lags 1/5/10/20 :101-104 · volume MA 20 :107 · chart window 20 :152 · train/val fraction 0.8 :243 · chart loop start index 20 :274.

**Seed:** none anywhere in the file.
**FS side effects:** `data/` and `data/images/charts/` mkdir **at import time** :32-33 · one PNG per (ticker,date) at :188-189, no existence check.

### 4.2 `sentiment.py` (163 lines)

For each `(date, ticker)` in `market_data` **from the last 30 calendar days only**, calls NewsAPI `/v2/everything` for a ±1-day headline window, scores up to 10 headline titles with FinBERT, and **replaces** the entire `sentiment_data` table.

**Functions** (no classes): `load_finbert()` :44 · `get_sentiment(texts, tokenizer, model) -> np.ndarray` :53 · `fetch_headlines(ticker, date, client) -> list` :75 · `run_sentiment_pipeline()` :98

**Entry point** `:162`. **No argparse** — no CLI surface at all.

**DB reads:** `SELECT DISTINCT date, ticker FROM market_data WHERE date >= date('now','-30 days') ORDER BY date DESC` :121-126.
**DB writes:** `CREATE TABLE IF NOT EXISTS sentiment_data (... PRIMARY KEY (date,ticker))` :105-116 · `to_sql("sentiment_data", if_exists="replace")` :152 — **drops the table and the PK just declared**. No `UPDATE` anywhere; the write is bulk.

**Row matching:** stores the raw `row["date"]` TEXT at :139 (not the `strftime` form built at :132, which is used only for the API call and log). This is why sentiment joins cleanly — see §5 D8.

**Constants:** `DB_PATH` :30 · `NEWS_API_KEY = os.getenv("NEWS_API_KEY")` :31 (name only; value present, redacted) · `MODEL_NAME = "ProsusAI/finbert"` :32 · `TICKER_KEYWORDS` :34-41. In-function: no-news default vector `np.array([0.0,1.0,0.0])` :56 · headlines per date `texts[:10]` :59 · `max_length=128` :64 · window ±1 day :79-80 · `page_size=10`, `sort_by="relevancy"` :85-89 · 30-day lookback :124 · score = `sentiment[2] - sentiment[0]` :144 · `time.sleep(0.2)` :149.

**Seed:** none. FinBERT inference is deterministic via `model.eval()` :48 and `torch.no_grad()` :67, but `sort_by="relevancy"` :88 makes the *input* headlines time-varying, so the module is not reproducible.
**FS side effects:** reads `.env` :20; writes to the HuggingFace cache :46-47.

### 4.3 `vision.py` (184 lines)

Loads every `market_data` row, renders a 20-day candlestick PNG when one is not already cached, pushes it through ImageNet EfficientNet-B0 with the classifier replaced by `Identity` to get a 1280-d float32 vector, and **replaces** `visual_features`.

**Functions** (no classes; no `nn.Module` subclass): `load_efficientnet()` :37 · `get_transform()` :49 · `generate_chart(df, date_idx, ticker) -> Path` :60 · `extract_features(image_path, model, transform) -> np.ndarray` :110 · `run_vision_pipeline()` :121

**Entry point** `:184`. **No argparse.**

**DB reads:** `SELECT * FROM market_data ORDER BY ticker, date` :141 (all 44 columns; only date/ticker/OHLCV used).
**DB writes:** `CREATE TABLE IF NOT EXISTS visual_features (... feature_vector BLOB, PRIMARY KEY (date,ticker))` :128-136 · `to_sql("visual_features", if_exists="replace")` :173 — **drops the PK and the BLOB declaration**; values remain BLOBs only because the sqlite3 driver binds `bytes` as BLOB.

**Constants:** `DB_PATH` :30 · `CHART_DIR = Path("data/images/charts")` :31 · `WINDOW = 20` :33 · `IMG_SIZE = 224` :34. In-function: `EfficientNet_B0_Weights.IMAGENET1K_V1` :39 · `classifier = Identity()` :43 · resize 224×224 :51 · ImageNet normalize :54-55 · batch dim `unsqueeze(0)` — **batch size 1 hardcoded** :113 · loop start index 20 :152 · logged dim `1280` :177 (a hardcoded string, not measured).

**Seed:** none. Determinism rests on `model.eval()` :44 and `torch.no_grad()` :115.
**Freezing:** the backbone is **not explicitly frozen** — no `requires_grad_(False)` anywhere. Safe here (one-shot extraction), unsafe if the object were handed to a trainer.
**FS side effects:** `CHART_DIR.mkdir` at import :32 · `path.exists()` short-circuit :87-88 · `savefig` :102 (no `dpi=`, unlike `ingest.py:189`).

### 4.4 `regime.py` (844 lines)

Fits a per-ticker 3-state Gaussian HMM on 4 price-derived features, writes `regime`/`regime_id` back into `market_data`, then runs a **fake numpy MC-Dropout over a placeholder model** and exports the resulting fabricated predictions to CSV.

**Functions/classes:** `build_hmm_features(df) -> np.ndarray` :80 · `@dataclass RegimeStats` :117 · `class MarketRegimeHMM` :128 (`__init__(n_regimes=N_REGIMES, random_state=42)` :137 · `fit(df)` :144 · `_build_state_map(df, features)` :166 · `label(df)` :197 · `regime_stats(df)` :210 · `save(path="models/hmm_model.pkl")` :249 · `load(path)` :254) · `class RegimeConditionedHead(nn.Module)` :267 (nested under `if TORCH_AVAILABLE:` :266; `forward(fused_repr, regime_ids)` :311) · `mc_dropout_predict(...)` :339 · `mc_dropout_numpy(pred_proba_fn, x, n_passes, dropout_rate, threshold)` :392 · `load_from_db(db_path, ticker)` :438 · `load_all_tickers(db_path)` :451 · `save_regimes_to_db(df, db_path, ticker)` :463 · `calibration_analysis(uncertainty, actual_correct, n_bins=10)` :527 · `run_regime_pipeline(...)` :564 · `_print_regime_stats` :684 · `_print_uncertainty_summary` :701 · `run_demo()` :725

**Entry point** `:808`. argparse :811-824: `--db` (default `"data/mmis.db"`) · `--ticker` · `--all` · `--demo` · `--no-save` · `--out` (default `"results"`).

**DB reads:** `SELECT * FROM market_data WHERE ticker=? ORDER BY date` :441-444 · `SELECT DISTINCT ticker` :454 · `SELECT date FROM market_data WHERE ticker=?` :489-491.
**DB writes:** `ALTER TABLE market_data ADD COLUMN regime TEXT / regime_id INTEGER` :480-485 · `UPDATE market_data SET regime=?, regime_id=? WHERE date=? AND ticker=?` via `executemany` :503-506, commit :507.

**Constants:** `DB_PATH = "data/mmis.db"` :60 · `N_REGIMES = 3` :61 · `HMM_ITERATIONS = 200` :62 · `MC_PASSES = 50` :63 · `MC_DROPOUT_RATE = 0.3` :64 · `UNCERTAINTY_THRESHOLD = 0.02` :65 · `MODEL_DIR = Path("models")` :66 · `REGIME_NAMES = {0:"mean_reverting",1:"trending",2:"high_vol"}` :69-73. In-function: `covariance_type="diag"` :153 · `tol=1e-4` :156 · rolling vol windows 5/20 :98-99.

**Seed:** `random_state=42` on the HMM :137 (hmmlearn only) · `np.random.default_rng(42)` :733 (**demo path only**). `mc_dropout_numpy` calls global `np.random.binomial` :409 with **no seed on the pipeline path** — the exported CSV is not reproducible.
**FS side effects:** `MODEL_DIR.mkdir` **at import** :67 · `models/hmm_{ticker}.pkl` :647-648 · `results/regime_stats_{ticker}.json` :654-657 · `results/regime_tagged_predictions.csv` :662,674 · `results/demo_regime_output.csv` :783-784.

### 4.5 `fusion.py` (569 lines)

Loads and left-joins the three tables, performs a date-sorted 80/20 temporal split, fits StandardScalers on train only, trains a cross-modal attention classifier with class-weighted CE and macro-F1 early stopping, fits a temperature scalar on val, and persists model + scalers + temperature.

**Functions/classes:** `set_seed(seed=SEED)` :46 · `class CrossModalAttention(nn.Module)` :74 (`__init__(market_dim, sentiment_dim, visual_dim, fusion_dim, num_heads, dropout)` :81 · `forward(market, sentiment, visual)` :138) · `class MultimodalDataset(Dataset)` :180 · `load_aligned_data(engine)` :199 · `prepare_features(df, market_scaler=None, visual_scaler=None)` :278 · `train_epoch(...)` :325 · `eval_epoch(...)` :349 · `collect_logits(...)` :374 · `fit_temperature(logits, labels) -> float` :387 · `run_fusion_pipeline()` :407

**Entry point** `:568`. **No argparse.** Only knob is env var `MMIS_SEED` :43.

**DB reads:** `market_data` — 35 columns (`date, ticker, target, regime` + 31 features) :205-216 · `sentiment_data` — `date, ticker, sentiment_neg, sentiment_neu, sentiment_pos` :219-223 · `visual_features` — `date, ticker, feature_vector` :226-229.
**DB writes: NONE.**

**Constants:** `SCALER_PATH = "models/feature_scalers.pkl"` :40 · `EARLY_STOP_PATIENCE = 7` :41 · `SEED = int(os.environ.get("MMIS_SEED","42"))` :43 · `DB_PATH` :57 · **`MARKET_DIM = 37`** :58 · `SENTIMENT_DIM = 3` :59 · `VISUAL_DIM = 1280` :60 · `FUSION_DIM = 256` :61 · `NUM_HEADS = 8` :62 · `NUM_CLASSES = 3` :63 · `BATCH_SIZE = 64` :64 · `EPOCHS = 30` :65 · `LR = 1e-4` :66 · `DROPOUT = 0.3` :67 · `MLFLOW_EXPERIMENT = "mmis_fusion"` :68 · `REGIME_ORDER = ["mean_reverting","trending","high_vol"]` :71 · `MARKET_COLS` (31 entries) :266-275. In-function: grad clip 1.0 :339 · LBFGS lr 0.01/max_iter 200 :395 · split fraction 0.8 :419 · `weight_decay=1e-4` :461 · sentiment fill `0.0/1.0/0.0` :245-247 · regime fill `"trending"` :251.

**Seed:** `set_seed()` :46-54, called at :408 — covers `random`, `np.random`, `torch`, `torch.cuda`, and `torch.use_deterministic_algorithms(True, warn_only=True)`. DataLoader generator seeded :439-440. **The only properly seeded module.**
**FS side effects:** `models/` mkdir :413 · `feature_scalers.pkl` written twice (:435-437 without temperature, :546-548 with) · `best_fusion_model.pt` :524 · MLflow run + `classification_report.txt` artifact :556 · `mlflow.pytorch.log_model` :559.

### 4.6 `inference.py` (252 lines)

Reloads the trained checkpoint plus persisted scalers/temperature, runs 50 batched MC-Dropout passes over the **entire** aligned dataset (train + val), fits an isotonic calibrator on the pre-val slice, and writes `results/final_predictions.csv`.

**Functions** (no classes): `mc_dropout_inference(model, market, sentiment, visual, n_passes=MC_PASSES, batch_size=256, device="cpu", temperature=1.0) -> dict` :55 · `run_inference()` :119

**Entry point** `:252`. **No argparse.**

**DB reads:** indirect — `create_engine(DB_PATH)` :126 → `load_aligned_data(engine)` :130, executing the three SELECTs in `fusion.py`. **No DB writes.**

**Constants:** `MC_PASSES = 50` :45 · `UNCERTAINTY_THRESHOLD = 0.02` :46 (**dead — never referenced again**) · `MODEL_PATH = "models/best_fusion_model.pt"` :47 · `CALIBRATOR_PATH = "models/calibrator.pkl"` :48 · `OUTPUT_CSV = "results/final_predictions.csv"` :49 · `UP_CLASS = 2` :52. In-function: `batch_size=256` :61 · uncertainty percentile 0.80 :197 · val-split quantile 0.80 :212 · `IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)` :215.

**Seed:** `np.random.seed(42)` :121 and `torch.manual_seed(42)` :122. Does **not** cover `torch.cuda.manual_seed_all`, and does not call `fusion.set_seed`.
**FS side effects:** reads `feature_scalers.pkl` :138-139 and `best_fusion_model.pt` :164 · writes `models/calibrator.pkl` :219-220 and `results/final_predictions.csv` :226.

### 4.7 `evaluation.py` (793 lines)

Consumes a predictions CSV/DataFrame and produces an equal-weight long-only backtest with turnover costs, three benchmarks, ECE calibration, per-ticker/per-regime/uncertainty breakdowns, a walk-forward block, and an out-of-sample block.

**Functions/classes:** `sharpe_ratio(returns, risk_free=0.0)` :53 · `max_drawdown(cumulative_returns)` :61 · `calmar_ratio(annualised_return, mdd)` :68 · `precision_at_up(pred_direction, actual_return)` :75 · `hit_rate(...)` :87 · `expected_calibration_error(pred_proba, actual_direction, n_bins=10)` :93 · `sortino_ratio(...)` :128 · `profit_factor(returns)` :137 · `@dataclass BacktestResult` :150 · `run_backtest(df, strategy_name="Model", use_uncertainty_filter=False, uncertainty_threshold=0.02)` :169 · `benchmark_buy_and_hold(df)` :265 · `benchmark_random(df, seed=42)` :273 · `benchmark_momentum(df, lookback=5)` :282 · `@dataclass WalkForwardFold` :300 · `walk_forward_split(df, train_months=6, test_months=1)` :312 · `class Evaluator` :377 (`__init__(predictions_df)` :393 · `full_report(...)` :405 · `equity_curves()` :577 · `print_summary(report)` :597 · `save_report(report, path)` :664 · `from_sqlite(db_path, predictions_table="predictions")` :673) · `_interpret_ece(ece)` :694 · `mock_predictions(n=500, seed=42)` :707

**Entry point** `:743`. argparse :746-759: `--db` · `--csv` · `--mock` · `--out` (default `"results/evaluation_report.json"`) · `--name` · `--uncertainty-filter`.

**DB reads:** `SELECT * FROM {predictions_table}` (default `"predictions"`) :684-686 — **no module in this repo ever creates a `predictions` table**, so `--db` is a dead path that raises. **No DB writes.**

**Constants:** `TRANSACTION_COST = 0.001` :44 · `SLIPPAGE = 0.0005` :45 · `TRADING_DAYS = 252` :46. In-function: `n_bins=10` :96 · default `uncertainty_threshold=0.02` :173,:410 · momentum lookback 5 :282 · `train_months=6, test_months=1` :314-315 · fold cap 24 :367 · quantile 0.80 :431,:504 · keep-fraction sweep `[0.5,0.7,0.8,0.9,1.0]` :479.

**Seed:** no global seed. Local only: `np.random.default_rng(seed=42)` :276 and :712.
**FS side effects:** JSON report written :666-668. No plots, no checkpoints.

### 4.8 Modules present but absent from the CONTEXT brief

`live_inference.py` (256 lines) — on-demand single-ticker serving; loads `hmm_*.pkl`, scalers, checkpoint, calibrator; hardcodes `market_dim=34` at :115-117. `gradcam.py` (107) — Grad-CAM over the vision backbone. `api\main.py` (99) — FastAPI wrapper, routes `/`, `/health`, `/analyze`, `/image/{filename}`. `dashboard\app.py` (109) — Streamlit UI over the API. `scripts\robustness_sweep.py` (63) — multi-seed sweep driver.

---

## 5. Defect verification

| # | Claim | Status | Evidence (file:line) | Notes |
|---|---|---|---|---|
| D1 | fusion.py uses a non-temporal train/val split (last tickers, not latest dates) | **DIFFERENT THAN DESCRIBED** | `fusion.py:260`, `fusion.py:418-420` | Fixed in `37682cc`. Explicit `sort_values(["date","ticker"])` precedes the split. Residual: one calendar date straddles the seam. |
| D2 | Scaler fit on train+val in fusion.py; inference.py refits a fresh scaler at serve time | **NOT FOUND** | `fusion.py:293-295,311-313,427-432`; `inference.py:134-144` | Both halves fixed. Fits are guarded by `is None`; scalers persisted and loaded. No `.fit_transform()` exists repo-wide. Residual: isotonic calibrator refits on an independently recomputed split. |
| D3 | regime.py exports a CSV of fake predictions from a numpy placeholder | **CONFIRMED** | `regime.py:612-626`, `:629-640`, `:662-677` | `dummy_model` feeds `mc_dropout_numpy`; output written to `results/regime_tagged_predictions.csv`. Logs at `:676-677` instruct feeding it to `evaluation.py`. |
| D4 | regime.py silent DB write; `WHERE date = ?` exact-string match can touch 0 rows, leaving regime NULL | **DIFFERENT THAN DESCRIBED** | `regime.py:488-508`, `:511-515`; DB: 0 NULL regime rows | The described bug was real and is fixed in `37682cc`. Dates resolved via a normalized-Timestamp lookup; rowcount checked via `conn.total_changes`; 0-row writes escalate to `logger.error`. Three residual weaknesses remain. |
| D5 | `RegimeConditionedHead` defined but never used; model uses a 3-dim regime one-hot | **CONFIRMED** | Definition `regime.py:267`; only other hits `regime.py:275`, `:354` (both docstrings); actual path `fusion.py:298-301` | Zero instantiations repo-wide. Its intended consumer `mc_dropout_predict` (`regime.py:339`) is also never called. |
| D6 | `MARKET_DIM = 37` while true dimensionality is 34 | **CONFIRMED** | `fusion.py:58`; `fusion.py:266-275`; `fusion.py:298-301` | True dim = 31 numeric + 3 one-hot = **34**. Constant is off by +3 **and is dead** — grep finds no reader. Runtime uses `.shape[1]`. |
| D7 | evaluation.py walk-forward loop: `train_start` never advances | **CONFIRMED** | `evaluation.py:333`, `:335-338`, `:363-364` | Assigned once at :333, never reassigned. All bounds recomputed from it each pass; `test_start = test_end` at :364 is dead. Produces 24 identical folds. |
| D8 | sentiment.py yields ~1.3% coverage; NewsAPI free tier returns ~30 days against a 2020→2026 range | **CONFIRMED** | `sentiment.py:124` (+ comment `:119`), `:152`; DB: 124/9,412 rows | Measured coverage **1.3175%**. All 124 rows fall in 2026-05-12 → 2026-06-10. |

### D1 — temporal split

```python
256	    # Sort by date so the downstream 80/20 split is a TRUE temporal split
260	    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
...
418	    # ── Temporal split BEFORE scaling (no train→val leakage) ────────
419	    split = int(len(df) * 0.8)
420	    train_df, val_df = df.iloc[:split], df.iloc[split:]
```

The claim describes the pre-`37682cc` code, which sliced already-scaled arrays with no sort. Today the sort key is `["date","ticker"]` with date primary, so `iloc[:split]` is a genuine chronological cut and no non-temporal path remains. **Residual:** with 6 tickers the split index rarely lands on a date boundary, so up to 5 rows of one calendar day sit in train while their siblings sit in val. Since `target` is the *next-day* direction (`ingest.py:135-140`), the last train date's label is drawn from the first val date — a one-day label overlap at the seam.

### D2 — scaler handling

```python
293	    if market_scaler is None:
294	        market_scaler = StandardScaler().fit(market_numeric)
295	    market_numeric = market_scaler.transform(market_numeric)
...
427	    # Fit scalers on TRAIN only, then transform both splits with them
429	     market_scaler, visual_scaler) = prepare_features(train_df)
430	    m_va, s_va, v_va, y_va, _, _ = prepare_features(
431	        val_df, market_scaler=market_scaler, visual_scaler=visual_scaler)
```

Both fits are guarded; train is fit-then-transform, val is transform-only. Scalers are pickled to `models/feature_scalers.pkl` (`fusion.py:435-436`, re-dumped with temperature at `:546-548`). `inference.py:134-144` raises if the file is missing and loads it; it never constructs a `StandardScaler`. `live_inference.py:109-111` and `:205-207` do the same. **Residual:** `inference.py:216` fits an isotonic calibrator whose `train_mask` derives from `out["date"].quantile(0.80)` (`:212-213`) — a different, independently recomputed boundary than `fusion.py:419`'s row-count split, computed after `:202` drops the last row per ticker. The two boundaries are approximately but not provably identical, so a handful of fusion-validation rows can leak into the calibrator fit and thus into the headline ECE.

### D3 — fabricated predictions

```python
612	        def dummy_model(x_dropped: np.ndarray) -> np.ndarray:
614	            Placeholder model: replace with real model.predict_proba() in Phase 4+.
615	            For now simulates a slight directional edge using log returns.
621	            up   = np.clip(0.35 + ret * 5, 0.05, 0.85)
622	            down = np.clip(0.35 - ret * 5, 0.05, 0.85)
...
674	        eval_df.to_csv(csv_path, index=False)
676	        logger.info(f"   Plug this directly into evaluation.py:")
677	        logger.info(f"   python evaluation.py --csv {csv_path} --uncertainty-filter")
```

Confirmed exactly as claimed. The synthetic probabilities flow through `mc_dropout_numpy` into `uncertainty`, `up_proba`, `pred_direction`, `high_uncertainty` (`regime.py:637-640`) and out to `results/regime_tagged_predictions.csv`. Nothing in the file marks it synthetic, and lines 676-677 actively instruct the operator to feed it to the evaluator — which would emit a fully formatted, entirely fictitious performance report. That file **is currently on disk** (1,231,157 B, see §6).

### D4 — the regime UPDATE

```python
489	    stored = cur.execute("SELECT date FROM market_data WHERE ticker=?", (ticker,)).fetchall()
492	    date_lookup = {pd.Timestamp(d[0]).normalize(): d[0] for d in stored}
496	        stored_date = date_lookup.get(pd.Timestamp(row["date"]).normalize())
497	        if stored_date is None:
498	            missing += 1
502	    before = conn.total_changes
503	    cur.executemany(
504	        "UPDATE market_data SET regime=?, regime_id=? WHERE date=? AND ticker=?", params)
507	    conn.commit()
508	    updated = conn.total_changes - before
```

The date parameter is **not** a naive `str()`/`strftime()` — it is resolved back to the exact stored string through a normalized-Timestamp lookup, and the rowcount **is** inspected (`:502`/`:508`), with 0-row writes escalated at `:511-515`. The commit message for `37682cc` confirms the original defect ("matched dates without microseconds, so every UPDATE silently hit 0 rows and the regime column was 100% NULL"). The database corroborates the fix: **0 rows have a NULL regime**. Three residual weaknesses: (a) `date_lookup` at `:492` is a dict keyed on normalized date, so if duplicate `(ticker,date)` rows at differing timestamps ever existed, all but the last would be silently skipped and counted in neither `updated` nor `missing`; (b) `missing` is logged but never escalated (`:518-519`) — a run where 90% of dates fail still logs a success; (c) a bare `except Exception: pass` at `:484-485` swallows any `ALTER TABLE` failure, not just "duplicate column".

### D5 — dead RegimeConditionedHead

Repo-wide grep for `RegimeConditionedHead` returns exactly three hits, all in `regime.py`, two inside docstrings: the definition at `:267`, a usage example in its own docstring at `:275`, and a parameter description at `:354`. No import, no instantiation, no reference from `fusion.py`, `inference.py`, or `live_inference.py`. What the trained model actually does:

```python
297	    # Regime one-hot (3 extra columns) — appended to market features (unscaled)
298	    regime_onehot = pd.get_dummies(df["regime"]).reindex(
299	        columns=REGIME_ORDER, fill_value=0).values.astype(np.float32)
301	    market_data = np.concatenate([market_numeric, regime_onehot], axis=1)
```

That vector enters a single shared `nn.Linear(market_dim, fusion_dim)` (`fusion.py:86-91`) and one classifier head for all regimes (`fusion.py:133`). There is no regime embedding, no per-regime weights, no routing. The "regime-conditioned output heads" feature documented at `regime.py:9` does not exist in the trained model — regime is three extra input features. Because the class is defined inside `if TORCH_AVAILABLE:` (`regime.py:266`), importing it when torch is absent raises `ImportError` rather than a clear message.

### D6 — MARKET_DIM

`fusion.py:58` reads `MARKET_DIM = 37        # Technical indicator features`. **Counting the actual columns:** `MARKET_COLS` at `fusion.py:266-275` contains 31 names — 4 (`rsi_14, macd, macd_signal, macd_diff`) + 5 (`ema_20, ema_50, bb_upper, bb_lower, bb_mid`) + 3 (`bb_bandwidth, bb_position, atr_14`) + 4 (`returns_1d/5d/10d/20d`) + 2 (`volume_ratio, volume_change_1d`) + 4 (`body_size, upper_shadow, lower_shadow, high_low_range`) + 4 (`day_of_year_sin/cos, month_sin/cos`) + 5 (`open, high, low, close, volume`) = **31**. Cross-checked against the database: of 35 candidate numeric columns in `market_data` (§3.8), `MARKET_COLS` uses 31 and omits `volume_ma_20`, `day_of_week`, `month`, `day_of_year`; nothing in `MARKET_COLS` is missing from the DB. Adding the 3-wide one-hot (`fusion.py:298-301`) gives **34**. The value 34 is corroborated by `results\inference_final.log`: "Market shape (incl. regime one-hot): (9292, 34)".

The constant is wrong by +3 **and is never read** — repo-wide grep for `MARKET_DIM` returns only the definition. `fusion.py:451` uses `market_dim=m_tr.shape[1]` and `inference.py:146` uses `market_data.shape[1]`, both dynamic and correct; `inference.py:26-37` pointedly omits `MARKET_DIM` from its import list. `live_inference.py:115-117` hardcodes the correct `market_dim=34` but is brittle to any `MARKET_COLS` edit. So D6 causes no runtime error today — it is a documentation trap for anyone who reads the constant as authoritative.

### D7 — walk-forward loop

```python
331	    folds = []
333	    train_start = min_date
335	    while True:
336	        train_end = train_start + pd.DateOffset(months=train_months)
337	        test_start = train_end
338	        test_end   = test_start + pd.DateOffset(months=test_months)
340	        if test_end > max_date:
341	            break
...
363	        # Expanding window: train_start stays fixed, only test window moves
364	        test_start = test_end
365	        fold_idx  += 1
367	        if fold_idx > 24:  # safety cap
368	            break
```

`train_start` is assigned once at `:333` and never reassigned. Because `train_end`, `test_start`, and `test_end` are all recomputed from it at the top of every iteration (`:336-338`), the assignment at `:364` is overwritten before use — dead code. The comment at `:363` describes behavior the code does not implement. The loop therefore emits **24 byte-identical folds**, terminated only by the `fold_idx > 24` cap, all testing the same month (months 6→7 after `min_date`, i.e. late 2020 — deep in-sample). This is directly visible in the artifact: all 24 fold objects in `results\evaluation_report.json` (lines 209-454) carry identical `train_start 2020-04-13`, `test_start 2020-10-13`, `sharpe 2.9229`, `precision_at_up 0.6267`, `n_test_days 138`. Two compounding problems: no retraining ever occurs (the function only slices a prediction column from a single already-fit model, so "walk-forward" is a misnomer), and the fold Sharpe at `:349` is computed on pooled ticker-days rather than portfolio-days, so `np.sqrt(252)` annualizes a series with ~6× the true observation count.

### D8 — sentiment coverage

```python
119	    # Load recent dates from market data (last 30 days — free API limit)
124	        WHERE date >= date('now', '-30 days')
...
152	    sentiment_df.to_sql("sentiment_data", con=engine, if_exists="replace", index=False)
```

Confirmed, with the free-tier constraint acknowledged in the code's own comment at `:119`. Measured from the database: **124 sentiment rows against 9,412 market rows = 1.3175%**, and every one of the 124 falls in 2026-05-12 → 2026-06-10 — a 30-day window against a market table spanning 2020-03-13 → 2026-06-10. Of those 124, only 100 carry a genuine reading (1.0625% of market rows); 24 are the no-news placeholder. `if_exists="replace"` at `:152` means repeated runs **cannot accumulate** coverage — each run's 30-day window overwrites the previous entirely. A further constraint the claim does not mention: the run issues ~126 requests against a free tier that allows 100/day, and over-quota failures are swallowed at `:93-95` into the same neutral placeholder.

### Unlisted defects found

**U1 — `market_data.regime` and `market_data.regime_id` are mutually inconsistent (HIGH).** The two columns are written together by one `UPDATE` (`regime.py:503-506`) but come from different value spaces:

```python
206	        df["regime_id"] = raw
207	        df["regime"]    = [self.state_map.get(int(s), "unknown") for s in raw]
```

`regime_id` is hmmlearn's **raw state index** (arbitrary, set by EM initialisation); `regime` is that index remapped through `state_map`, built by **sorting states by realised volatility** (`regime.py:188-194`). A separate HMM is fit per ticker (`regime.py:595-601`), so the raw→name permutation is redrawn for each one. Measured: the mapping is 1:1 *within* each ticker but permuted *across* them — AAPL/AMZN/MSFT/TSLA map mean_reverting→0, GOOGL maps mean_reverting→1/trending→0, SPY maps mean_reverting→2/trending→0/high_vol→1. Only **6,666 / 9,412 rows (70.82%)** satisfy the contract declared in `REGIME_NAMES` (`regime.py:69-73`), which is never used to derive `regime_id` (its sole reference is a stats loop at `:219`). The trained model is unaffected — `fusion.py:298-300` one-hots the `regime` *string* — but any consumer grouping or joining on `regime_id` silently merges three different semantic regimes. `RegimeStats.regime_id` (`regime.py:238`) propagates the raw index into every `results/regime_stats_*.json`, making those files cross-ticker incomparable.

**U2 — FinBERT label order is inverted (HIGH).** `sentiment.py:54,56,141-144` assume logit order `[negative, neutral, positive]`. The `ProsusAI/finbert` config declares `{0:"positive", 1:"negative", 2:"neutral"}`. So `sentiment_neg` holds P(positive), `sentiment_neu` holds P(negative), `sentiment_pos` holds P(neutral), and `sentiment_score = sentiment[2] - sentiment[0]` (`:144`) is P(neutral) − P(positive) — not a polarity score. The no-news default `[0.0, 1.0, 0.0]` (`:56`) is, under the true order, P(negative) = 1.0 — a maximally bearish vector, not neutral. Impact is currently masked by the 1.3% coverage.

**U3 — the last row per ticker carries a fabricated label (MEDIUM).** `ingest.py:135-140` computes `next_return` as a bare Series never assigned into `df`. On the final row `close.shift(-1)` is NaN, both `.loc` masks are False, and `target` keeps its default `1` (Flat). `df.dropna()` at `:143` cannot remove the row because no *column* is NaN there. The comment at `:142` claiming the last row is dropped is false. Six synthetic labels, always at the newest date — the rows live inference touches first.

**U4 — the regime feature carries look-ahead bias (HIGH).** `regime.py:105-108` standardizes HMM observations with full-sample mean/std; `:151-158` fits on the complete series; `:205` runs a Viterbi decode over the whole sequence, so the state at day *t* is conditioned on observations at *t+1 … T*; `:188-194` then names states by full-sample volatility ranking. Every training row's `regime` therefore encodes future information, and this leaks across the train/val boundary regardless of how cleanly D1's split is done. README lines 130-133 acknowledge this.

**U5 — the uncertainty filter is a no-op in the OOS path (MEDIUM).** `evaluation.py:429-439` derives a percentile threshold for the full-sample backtest, but `:508-510` passes the raw default `uncertainty_threshold=0.02` (`:410`) to the OOS backtest. Measured MC-Dropout uncertainty maxes at 0.00375 (§6), so `uncertainty <= 0.02` is true for every row. The line printed at `:614-620` as "← THE HONEST HEADLINE" is unfiltered while the full-sample line is filtered, yet both are labelled the same strategy.

**U6 — training-label and evaluation-metric definitions disagree (MEDIUM).** The UP class requires `next_return > 0.005` (`ingest.py:139`), but `precision_at_up` scores success as `actual_return > 0` (`evaluation.py:83`) and `hit_rate` likewise (`:89`). The model is trained to predict a >0.5% move and graded on a >0% move; the two disagree on every 0–0.5% day. The `base_up_rate` comparison at `:512` inherits the same mismatch, so the reported "precision edge in pts" is measured against the wrong null.

**U7 — ECE silently drops the most overconfident predictions (MEDIUM).** `evaluation.py:111` bins with `(pred_proba >= lo) & (pred_proba < hi)`; the final bin's `hi` is exactly 1.0, so `pred_proba == 1.0` matches no bin, while `n = len(pred_proba)` at `:107` still counts it in the denominator. `inference.py:215` builds `IsotonicRegression(out_of_bounds="clip", y_max=1.0)`, which emits exactly-1.0 routinely (2 such rows in the current CSV). Reported ECE is biased low precisely on the predictions calibration error exists to catch.

**U8 — `ingest.py` is not idempotent (MEDIUM).** `to_sql(..., if_exists="append")` (`:251`) against a table with `PRIMARY KEY (date,ticker)` (`:220`) raises `IntegrityError` on any re-run; the per-ticker `except Exception` at `:284-286` swallows it and logs `❌ Failed for {ticker}`. Refreshing data requires manually dropping the table. Note this also means the duplicate-row hazard behind `fusion.py:204`'s "duplicates removed" comment (which performs no dedupe) is currently unrealized — the DB has 0 duplicates.

**U9 — two divergent chart renderers write the same filenames (MEDIUM).** `ingest.generate_chart(df, ticker, date_idx, ...)` (`:151`) and `vision.generate_chart(df, date_idx, ticker)` (`:60`) have the 2nd and 3rd positional parameters **swapped**, write to the same `{TICKER}_{DATE}.png` pattern, and use different plot parameters (`ingest.py:183,189` pass `scale_padding` and `dpi=100`; `vision.py:91-102` pass neither, and `:75-81` conditionally omit the MA overlays). `vision.py:87-88` short-circuits on `path.exists()`, so whichever ran first wins. The feature extractor's input distribution depends on run history, and nothing in the DB records which renderer produced an image.

**U10 — `evaluation.py --db` is a dead path (LOW).** `from_sqlite` (`:673-687`) reads a `predictions` table. Repo-wide, the only tables ever created are `market_data`, `chart_images`, `visual_features`, `sentiment_data`. Invoking `--db` raises.

**U11 — `pred_direction` is not derivable from `pred_proba` (HIGH, artifact-level).** In `results\final_predictions.csv`, `pred_direction == (pred_proba > 0.5)` for only **37.06%** of rows. `pred_direction` is the 3-class argmax; `pred_proba` is the calibrated binary P(up). Every return metric is driven by `pred_direction`, while the advertised ECE is measured on `pred_proba` — **the calibration number describes a probability that does not drive the trades**. Compounding this, `pred_proba_uncalibrated` never exceeds 0.5228, so the raw head essentially never says "up" on a >0.5 basis; all UP trades come from the argmax path.

**U12 — the regime layer is 100% dead at serve time (HIGH).** `results\api_server.log` records that all 9 logged `/analyze` requests emitted `Regime labelling failed — Can't get attribute 'MarketRegimeHMM' on <module 'uvicorn.__main__'>`, after which the handler defaults every prediction to `regime='trending'`. The `hmm_*.pkl` files were pickled from `__main__` and cannot be unpickled under uvicorn. Since the regime one-hot is 3 of the 34 market dims, this is a hard train/serve skew. README lines 160-162 claim the opposite.

**U13 — `regime_stats` never counts the final run (LOW).** `regime.py:226-235` appends `cur_len` only on a transition, so the run in progress at loop exit is never counted; `avg_duration_days` systematically omits the last run of every regime. If `runs` ends up empty, `:235` falls back to `float(mask.sum())` — a total day count, not a duration. Visible in the artifacts: AAPL and GOOGL report `avg_duration_days: 1.0` for two of three regimes.

**U14 — `_build_state_map` hardcodes 3 states (LOW).** `regime.py:190-194` indexes `sorted_by_vol[0..2]` despite `n_regimes` being a constructor parameter (`:137`); `n_regimes=2` raises `IndexError`.

**U15 — unknown regime strings become a silent all-zero one-hot (LOW).** `MarketRegimeHMM.label` can emit `"unknown"` (`regime.py:207`); `fusion.py:298-300` reindexes onto `REGIME_ORDER`, dropping it without warning and creating an implicit fourth regime encoded `[0,0,0]`.

**U16 — `torch.load` without `weights_only=True`** at `fusion.py:536`, `inference.py:164`, `live_inference.py:119` (LOW, given checkpoints are locally produced).

---

## 6. Artifacts and prior results

### `results\` — 39 files, 2.31 MB (4 CSV, 7 JSON, 28 LOG; no PNG, no MD)

| File | Bytes | Modified |
|---|---|---|
| `final_predictions.csv` | 984,862 | 2026-06-15 20:16:49 |
| `regime_tagged_predictions.csv` | 1,231,157 | 2026-06-14 20:42:23 |
| `demo_regime_output.csv` | 92,600 | 2026-06-11 14:55:16 |
| `robustness_sweep.csv` | 329 | 2026-06-15 20:16:51 |
| `evaluation_report.json` | 11,904 | 2026-06-15 20:16:51 |
| `regime_stats_{AAPL,AMZN,GOOGL,MSFT,SPY,TSLA}.json` | 605–608 each | 2026-06-14 20:42:22-23 |
| `api_server.log` | 15,947 | 2026-06-16 00:07:38 |
| `streamlit.log` | 3,950 | 2026-06-16 00:07:38 |
| `eval_{new,v3,fixed,final}.log` | 1,966–2,443 | 2026-06-14 20:49 → 06-15 20:11 |
| `fusion_{retrain,retrain_v2,v3,final}.log` | 498–4,508 | 2026-06-14 20:46 → 21:08 |
| `inference_{new,v2,v3,fixed,final}.log` | 995–2,865 | 2026-06-14 20:48 → 06-15 20:11 |
| `sweep_{fusion,infer,eval}_{0,7,42,123}.log` | 2,429–4,354 | 2026-06-15 20:13 → 20:16 |
| `robustness_sweep.log` | 1,309 | 2026-06-15 20:16:51 |

**CSV row counts and columns**

| CSV | Rows | Columns |
|---|---|---|
| `final_predictions.csv` | 9,286 | `date, ticker, close, regime, pred_proba, pred_direction, uncertainty, actual_return, high_uncertainty, pred_proba_uncalibrated` |
| `regime_tagged_predictions.csv` | 9,406 | `date, ticker, close, regime, uncertainty, up_proba, pred_direction, high_uncertainty, actual_return, pred_proba` |
| `demo_regime_output.csv` | 600 | `date, ticker, open, high, low, close, volume, regime_id, regime, uncertainty, up_proba, pred_direction, high_uncertainty` |
| `robustness_sweep.csv` | 4 | `seed, oos_precUP, oos_base_up, oos_edge_pts, oos_sharpe, oos_bh_sharpe, oos_ece, full_precUP, full_sharpe` |

**`results\final_predictions.csv` detail.** 9,286 rows, 0 nulls, 0 duplicate `(date,ticker)`. Date range **2020-04-13 → 2026-06-09**. Date format is bare ISO `YYYY-MM-DD` with **no time component** — three raw values: `2020-04-13`, `2020-04-14`, `2020-04-15`. **This differs from the database's 26-char format** (§3.4). Distinct tickers (6): AAPL 1,547 · AMZN 1,548 · GOOGL 1,547 · MSFT 1,548 · SPY 1,548 · TSLA 1,548. `pred_direction`: **0 → 5,847 (62.97%), 1 → 3,439 (37.03%)** — binary only; no `2` is ever emitted despite the 3-class head. `pred_proba` has only 15 distinct values (isotonic step function), min 0.0, max 1.0. `pred_proba_uncalibrated` max is 0.5228. `uncertainty` ranges 0.000484 → 0.00375.

### Staleness relative to producing code

Producer mtimes: `regime.py` 06-14 20:41 · `fusion.py` 06-15 00:12 · `evaluation.py` 06-15 15:56 · `inference.py` 06-15 19:51 · `live_inference.py` 06-15 21:50.

**The four load-bearing artifacts are CURRENT (not stale):** `final_predictions.csv` (NEWER than `inference.py` by 25m), `evaluation_report.json` (NEWER than `evaluation.py` by 4h20m), `robustness_sweep.csv` (NEWER), `eval_final.log` (NEWER). All were produced by the seed-42 leg of the robustness sweep on 2026-06-15 20:13–20:16.

**17 of 39 results files are STALE** — all logs plus two CSVs:

- `regime_tagged_predictions.csv` — NEWER than `regime.py` (+62s) but **OLDER than `fusion.py`, `inference.py`, and `evaluation.py`** → stale relative to the current model.
- `demo_regime_output.csv` — 3 days older than `regime.py` → abandoned.
- `fusion_final.log`, `inference_final.log`, `eval_fixed.log`, `inference_fixed.log`, and all `*_new` / `*_v2` / `*_v3` logs → older than their producers. Note three files named `*_final.log` are **not** final: `fusion_final.log` predates `fusion.py`, and `inference_final.log` predates the run that actually wrote `final_predictions.csv`.

The database itself (mtime 2026-06-14 20:42:23, the `regime.py` write) predates every model and prediction artifact, so all of them derive from one consistent DB snapshot.

### Model checkpoints (`models\`, 10 files — none loaded)

| Path | Bytes | Modified | Type |
|---|---|---|---|
| `models\best_fusion_model.pt` | 4,396,394 | 2026-06-15 20:15:59 | PyTorch checkpoint |
| `models\best_fusion_model.PRE_REGIME_FIX.pt` | 4,396,394 | 2026-06-14 20:45:03 | PyTorch checkpoint — manual backup; same size, different SHA-256 |
| `models\feature_scalers.pkl` | 32,116 | 2026-06-15 20:16:14 | Pickle — StandardScalers + temperature |
| `models\calibrator.pkl` | 656 | 2026-06-15 20:16:49 | Pickle — isotonic calibrator |
| `models\hmm_{AAPL,AMZN,GOOGL,MSFT,SPY,TSLA}.pkl` | 1,511–2,618 | 2026-06-14 20:42:22-23 | Pickle — per-ticker HMM |

The serving triple (`best_fusion_model.pt`, `feature_scalers.pkl`, `calibrator.pkl`) is mutually consistent — all three from the seed-42 sweep leg. The six HMM pickles are 24 h older and, per U12, **cannot be unpickled by the API process at all**.

### Chart images

One image directory: `data\images\charts` — **9,310 PNG, 47.53 MB**. Naming `{TICKER}_{YYYY-MM-DD}.png` (9,305 files, e.g. `AAPL_2020-04-13.png`) plus `{TICKER}_{YYYY-MM-DD}_gradcam.png` (5 files, from live-demo runs on 2026-06-16). Per-ticker: 1,552 each except SPY 1,550.

### MLflow

`mlruns\` on disk is **not a valid file-store** — 0 `meta.yaml`, 0 `metrics\` directories. It holds artifacts only: `mlruns\1\` contains 16 run directories (each just `artifacts\classification_report.txt`, 389 B) plus `mlruns\1\models\` with 16 logged-model dirs each carrying a 4.4 MB `model.pth` — 67 MB of near-duplicate checkpoints.

The real backing store is **`mlflow.db`** (46 tables, read via read-only URI):

- **Experiments: 2** — `0 / Default` (0 runs; its artifact dir `mlruns\0` does not exist) and `1 / mmis_fusion` (**17 runs**).
- **Runs: 17** — 16 `FINISHED`, 1 `RUNNING` (started 2026-06-12 10:54:20, never closed, no directory on disk — orphan; this explains the 17-vs-16 gap).
- Run start times: earliest 2026-06-11 15:56:38, latest 2026-06-15 20:15:51. **Nothing covers the Layer-6 demo work of 06-15 21:50 → 06-16 00:07.**
- **Metric names (8):** `best_val_acc`, `best_val_macro_f1`, `calibration_temperature`, `train_acc`, `train_loss`, `val_acc`, `val_loss`, `val_macro_f1`.
- **Param names (12):** `batch_size`, `class_weighted_loss`, `dropout`, `early_stop_patience`, `epochs`, `fusion_dim`, `lr`, `market_dim`, `model_selection`, `num_heads`, `regime_conditioning`, `scaler_fit`.
- **No trading or evaluation metric is tracked** — no sharpe, precision_at_up, ECE, or return key exists. Every advertised number lives only in loose files.

### Tests and CI

**There are none.** Zero `test_*.py` / `*_test.py`, no `tests\` directory, no `conftest.py`, no `pytest.ini` / `tox.ini` / `setup.cfg` / `pyproject.toml`, no `.github\`, no `.gitlab-ci.yml`, no `Makefile`. No test was executed (none exists to execute). Every claimed bug fix in the README is unguarded by a regression test.

### Contradictions between artifacts

- **`regime_tagged_predictions.csv` vs `final_predictions.csv`:** merging on `(date,ticker)` gives 9,286 matched rows. `close`, `actual_return`, and `regime` are 100% identical, but **`pred_direction` agrees on only 44.16%** of rows, `high_uncertainty` is 89.5% flagged in the old file vs 20.0% in the new, and the `uncertainty` scale differs ~19× (old mean 0.0248 vs new 0.00131). The old file is fabricated output (D3) and carries no deprecation marker.
- **`evaluation_report.json` headline block is internally inconsistent:** line 21 `n_trades: 3283` comes from the 80%-coverage filter row (line 191), but line 18 `precision_at_up: 0.5787` equals the 100%-coverage value (line 204), and line 14 `sharpe: 1.896` matches neither filter row (80% → 1.9383, 100% → 1.9765). **CANNOT DETERMINE** which coverage the advertised 1.896 Sharpe belongs to.
- **Log generations disagree wildly and all remain on disk:** `eval_new.log` total return −99.35% / ECE 0.11746 · `eval_v3.log` −11.22% / ECE 0.18139 · `eval_fixed.log` +466.94% / ECE 0.19512 · `eval_final.log` +466.94% / ECE 0.00403.
- **Cardinality drift, unexplained:** `market_data` 9,412 → `visual_features` 9,292 → aligned 9,292 → `final_predictions.csv` 9,286 → chart PNGs 9,310 → `regime_tagged_predictions.csv` 9,406. Six different counts; the 9,292 → 9,286 drop is not logged anywhere.
- **Robustness sweep contradicts the README's chosen headline:** across seeds 0/7/42/123, OOS precision edge > 0 in **0/4 seeds** and OOS beats buy-and-hold Sharpe in **0/4 seeds**. Full-sample Sharpe ranges 1.312 → 1.896 with mean 1.611; **seed 42 is the best of the four on both `full_sharpe` and `oos_sharpe`, and is the seed the README headlines.**
- **README line 11 is unsupported by any current artifact:** it states "walk-forward precision is ~50% (coin-flip)", but `evaluation_report.json:453` reports `mean_precision_at_up: 0.6267` and `:452` `mean_sharpe: 2.9229`. Those figures come from the degenerate 24-identical-fold loop (D7), so neither the README's number nor the artifact's is meaningful — but they do not match each other.

---

## 7. Discrepancies versus the claimed state

1. **Module count.** CONTEXT describes seven modules. The repo contains **thirteen** Python files, including five not mentioned: `live_inference.py` (256), `gradcam.py` (107), `api\main.py` (99), `dashboard\app.py` (109), `scripts\robustness_sweep.py` (63). All seven named modules exist at the claimed root paths.
2. **"~31 technical indicators."** Ambiguous and resolvable three ways. `add_indicators` adds **33** columns (`ingest.py:64-148`); 31 of those are non-label/non-ID; genuine technical/price-derived indicators number **23**, plus 7 calendar features. The DB holds **44** columns of which **35** are candidate numeric features. The code's own console banner prints **37** (`ingest.py:296`). "~31" matches exactly one of these definitions.
3. **"1280-dim features stored as frozen BLOBs."** Dimension confirmed (5,120 bytes = 1,280 float32, uniform). "Frozen" is imprecise: `vision.py` never calls `requires_grad_(False)`; freezing is effected only by `model.eval()` (`:44`) and a locally scoped `torch.no_grad()` (`:115`). The column is also declared `TEXT`, not `BLOB` (§3.1).
4. **"MC-Dropout (50 passes) → results\final_predictions.csv."** Confirmed (`inference.py:45`, `:226`). But `regime.py` contains a **second, fake** MC-Dropout path (`mc_dropout_numpy`, `:392`) over a placeholder model, which writes a separate CSV — D3.
5. **"Cross-modal attention (market queries sentiment + vision) → 3-class head."** Architecturally confirmed. In practice the sentiment modality is a near-constant `[0,1,0]` for ~98.7% of rows (`fusion.py:245-247` imputes it), so cross-modal attention over sentiment is attending to a constant. The 3-class head also never emits class 2 in the current predictions file — `pred_direction` is binary (§6).
6. **"Gaussian HMM → 3 regimes."** Confirmed, but the HMM is fit **per ticker** (`regime.py:595-601`), producing six independent models whose state numbering is not aligned — U1.
7. **Documented run order.** `ingest → sentiment → vision → regime → fusion → inference → evaluation` **holds**, and the read/write map in §4 confirms it, with two refinements:
   - `sentiment.py` and `vision.py` are **mutually independent** — they touch disjoint tables (`sentiment_data`, `visual_features`) and both only read `market_data`. Their relative order is arbitrary; the documented sequence over-constrains.
   - `regime.py` **must** precede `fusion.py`: `fusion.py:206` selects `regime`, and `fusion.py:251` would otherwise fill every row with the constant `"trending"`. This is a hard dependency the CONTEXT list happens to satisfy but does not state.
   - Additionally, `evaluation.py` must be invoked with `--csv`; its `--db` path is dead (U10). And `inference.py` must run before `live_inference.py`/the API, which need `models/calibrator.pkl`.
8. **Defect claims D1, D2, D4 describe code that has since been fixed.** D1 and D2 were repaired in `37682cc`; D4's exact-string date bug was also repaired in `37682cc`, and the DB shows 0 NULL regime rows. The CONTEXT description of the current state is out of date on these three.
9. **D6 is real but inert.** `MARKET_DIM = 37` is wrong (true value 34), but the constant is read nowhere, so it produces no runtime error — the practical risk is documentation, not behavior.
10. **Config key name.** CONTEXT references `NEWSAPI_KEY`; the actual key in `.env` and at `sentiment.py:31` is **`NEWS_API_KEY`**.
11. **`chart_images` table.** Created by `ingest.py:226-234` with a foreign key, referenced in the schema, and **never written to by any module** — 0 rows. The pipeline stores image paths in `visual_features.image_path` instead.
12. **No virtual environment and no tests.** Neither is claimed in CONTEXT, but both bear on "already built and run end to end": the environment is a 152-package system-wide Python, and there is zero automated verification.
13. **Nothing that produces the advertised numbers is version-controlled.** `.gitignore` excludes `results/`, `models/*.pt`, `models/*.pkl`, `mlruns/`, `mlflow.db`, `data/mmis.db`, `data/images/`. `git ls-files` returns 17 source files. A fresh clone reproduces none of §3 or §6, and MLflow records no evaluation metric, so there is no versioned record of any advertised result.
14. **`evaluation.py` self-identifies as a different module.** Its docstring at `:2` calls it `evaluate.py` and `:13` documents `from evaluate import Evaluator` — a module that does not exist.

---

## 8. Open questions

1. **Which uncertainty coverage does the advertised Sharpe of 1.896 correspond to?** `evaluation_report.json` line 14 matches neither the 80% (1.9383) nor the 100% (1.9765) filter row, and the headline block mixes filtered `n_trades` with unfiltered `precision_at_up`. Resolving this requires re-running `evaluation.py` under known flags.
2. **Why do `final_predictions.csv` (9,286) and the aligned dataset (9,292) differ by 6 rows?** The drop is not logged. Suspected to be the per-ticker last-row `dropna` at `inference.py:202`, but 6 ≠ 6 tickers × 1 exactly matches only if every ticker loses exactly one — unverified without running.
3. **Do the 9,310 chart PNGs correspond 1:1 to any table?** Three cardinalities (9,310 / 9,292 / 9,286) and two renderers (U9) mean the provenance of individual images cannot be determined from disk state alone.
4. **Which renderer produced the images currently on disk?** `ingest.py` and `vision.py` write the same filenames with different plot parameters, and `vision.py:87-88` skips existing files. Nothing recorded in the DB distinguishes them. Determining this would require pixel inspection against a freshly rendered reference.
5. **Is `best_fusion_model.PRE_REGIME_FIX.pt` still needed?** Same size, different SHA-256 from the live checkpoint, referenced by no code path. Its retention intent is a human question.
6. **What is the orphaned MLflow run `925de685…`?** Status `RUNNING` since 2026-06-12, 68 metric rows in `mlflow.db`, no directory in `mlruns\1\`. Cannot tell whether it was a crashed run or an aborted experiment.
7. **Was the FinBERT label-order bug (U2) ever intentional?** The code is internally consistent with a `[neg,neu,pos]` assumption; whether the author verified against the model card is unknowable from the repo.
8. **Does the ~2-month gap since the last commit (2026-06-15 → today) reflect abandonment or a pause?** No repo evidence either way.
9. **Which of the 24 log-file generations corresponds to the committed README numbers?** Only reconcilable by mtime correlation; the `_new`/`_v2`/`_v3`/`_fixed`/`_final`/`sweep_*` naming carries no ordering guarantee, and three `*_final.log` files predate their producers.

---

## 9. Top risks before any edit

1. **Re-running `ingest.py` will not refresh data and will fail silently per ticker.** `to_sql(if_exists="append")` against `PRIMARY KEY (date,ticker)` raises `IntegrityError`, swallowed at `ingest.py:284-286`. Worse, `END_DATE = datetime.today()` (`:26`) means a re-run *today* would attempt to append ~2 months of new rows and abort the whole per-ticker transaction — leaving the DB in its current state while logging failure. Any data refresh must be planned as an explicit migration, not a re-run.

2. **Re-running `sentiment.py` or `vision.py` destroys existing data.** Both use `to_sql(if_exists="replace")` (`sentiment.py:152`, `vision.py:173`), which DROPs the table. For sentiment this is irreversible: the current 124 rows cover 2026-05-12 → 2026-06-10, and NewsAPI's free tier can no longer return that window from today's date — **a re-run would replace real sentiment with a fresh 30-day window or, if the quota/key fails, with 100% neutral placeholders.** The only copy of that data is in `data\mmis.db`, which is gitignored.

3. **Fake predictions are on disk, undistinguished from real ones, and the tooling invites their use.** `results\regime_tagged_predictions.csv` (1.2 MB) is `dummy_model` output (D3), shares 8 of 10 column names with the real file, and `regime.py:676-677` prints the exact `evaluation.py --csv` command to feed it in. Running that produces a fully formatted, entirely fictitious report. Any change to the evaluation flow risks picking up this file.

4. **`regime_id` is corrupt across tickers and any analysis keyed on it is wrong (U1).** Only 70.82% of rows honour the declared `REGIME_NAMES` contract. Code that switches from the `regime` string to the `regime_id` integer — an obvious-looking optimization — silently merges three different semantic regimes. The six `results\regime_stats_*.json` files already propagate the raw index.

5. **Reported metrics rest on a hand-picked seed and a broken walk-forward.** Seed 42 is the best of four on both `full_sharpe` and `oos_sharpe`, and it is the seed the README headlines; the sweep shows OOS edge > 0 in **0/4 seeds**. The 24-fold walk-forward is one in-sample month repeated 24 times (D7). Changing any of `inference.py`, `evaluation.py`, or the split logic will move these numbers, and there is no baseline test to say whether the movement is a fix or a regression.

6. **There are zero tests and zero CI.** No `tests\`, no `.github\`, no `pytest.ini`. Every fix listed in the README — including the three already-repaired defects D1/D2/D4 — is unguarded. Any edit is unverifiable except by re-running the full pipeline, which per risks 1 and 2 is itself destructive.

7. **The serving path is already broken and the calibration number is measured on the wrong column.** The API defaults every prediction to `regime='trending'` because the HMM pickles cannot be unpickled under uvicorn (U12), so 3 of 34 market dims are wrong at serve time. Separately, `pred_direction` (which drives all returns) agrees with `pred_proba > 0.5` (which the ECE measures) on only 37.06% of rows (U11) — so "ECE 0.004" does not describe the traded signal. Both must be understood before any serving or calibration change.
