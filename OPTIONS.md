# Options VRP Trading Pipeline

## Overview

The **Variance Risk Premium (VRP)** is the persistent spread between implied volatility (what the market prices in) and realized volatility (what actually happens). Because option sellers are compensated for bearing tail risk, IV systematically overstates RV, creating a harvestable premium.

This pipeline implements a signal-driven options selling system on NSE F&O. It computes vol signals (IV/RV ratio, IV percentile rank, term structure), applies rule-based and optional ML entry filters, constructs multi-leg positions (straddles, iron butterflies), backtests across expiry cycles, and executes live with greeks-based risk management.

## Quick Start

```bash
# Backtest a short straddle on NIFTY
python main.py options backtest --strategy vrp_straddle --underlying NIFTY --start 2022-01-01 --end 2024-12-31

# Backtest iron butterfly with custom capital
python main.py options backtest --strategy vrp_butterfly --underlying BANKNIFTY --start 2023-01-01 --end 2024-12-31 --capital 500000

# Vol research analytics
python main.py options research --underlying NIFTY --start 2023-01-01 --end 2024-12-31

# Screen F&O universe by liquidity
python main.py options screener --min-volume 1000 --min-oi 5000

# Train ML model
python main.py options train --underlying NIFTY --model-type regression --output models/vrp_nifty.pkl

# Live trading (dry-run by default)
python main.py options live --underlyings NIFTY,BANKNIFTY --strategy vrp_straddle --dry-run

# Live trading (real orders)
python main.py options live --underlyings NIFTY --strategy vrp_straddle --no-dry-run
```

## Architecture

### Package Layout

```
config/
    options_config.py            # OptionsConfig, VRPConfig, ModelConfig
data/
    chain.py                     # OptionChain, OptionContract, fetch_option_chain, fetch_expiries
    iv_engine.py                 # BS pricing, Newton-Raphson IV solver, realized vol
    fno_historical.py            # 7-day paginated F&O candle fetcher
    vol_surface.py               # vol cone, term structure, flat forward ratio
    universe.py                  # F&O liquidity screener
signals/
    base.py                      # OptionType, StrategyAction, OptionLeg, OptionsSignal
    ivrv.py                      # IV/RV ratio signal
    iv_percentile.py             # Rolling IV percentile rank
    flat_forward.py              # Near/far term structure ratio
    composite.py                 # VolSignalSnapshot + is_vrp_entry()
strategies/
    options_base.py              # OptionsStrategy ABC
    vrp_straddle.py              # Short ATM straddle
    vrp_butterfly.py             # Iron butterfly
models/
    features.py                  # Feature matrix from VolSignalSnapshots
    pipeline.py                  # sklearn GradientBoosting + TimeSeriesSplit
    predict.py                   # Live inference from saved model
backtesting/
    options_engine.py            # Expiry-cycle backtest engine
    options_pnl.py               # Multi-leg P&L calculator
    options_report.py            # Stats table + equity curve + vol cone charts
execution/
    options_broker.py            # Per-leg FNO order placement
    options_safeguards.py        # Greeks-based risk limits
    delta_hedger.py              # Periodic delta neutralization
runner/
    options_cli.py               # Click CLI subgroup (backtest, research, screener, train, live)
    live_vrp.py                  # LiveVRPRunner polling loop
```

### Data Flow

```
CLI (runner/options_cli.py)
 |
 v
Engine (backtesting/options_engine.py / runner/live_vrp.py)
 |
 +---> Data Layer: fetch_option_chain() + fetch_historical() + realized_vol()
 |         |
 |         v
 +---> Signals: compute_snapshot() --> VolSignalSnapshot
 |         |
 |         v
 +---> Entry Filter: is_vrp_entry() + optional model predict_vrp()
 |         |
 |         v
 +---> Strategy: generate_signal() --> OptionsSignal with legs
 |         |
 |         v
 +---> Safeguards: check greeks exposure, trading hours, daily loss
 |         |
 |         v
 +---> Execution: OptionsBroker.place_legs() per leg
 |         |
 |         v
 +---> Position Mgmt: exit_condition() + DeltaHedger.check_and_hedge()
```

## Modules

### Config (`config/options_config.py`)

Three dataclass configs, loaded as lazy properties on `Settings`:

**OptionsConfig** - General options trading parameters:

| Field | Type | Default | Description |
|---|---|---|---|
| `risk_free_rate` | `float` | `0.065` | India 10Y rate for BS pricing |
| `iv_lookback_days` | `int` | `252` | Historical IV lookback window |
| `rv_window` | `int` | `30` | Realized vol rolling window (trading days) |
| `min_option_volume` | `int` | `1000` | Minimum contract volume filter |
| `min_oi` | `int` | `5000` | Minimum open interest filter |
| `days_before_expiry_entry` | `int` | `7` | Enter N days before expiry |
| `days_before_expiry_exit` | `int` | `1` | Mandatory exit N days before expiry |

**VRPConfig** - VRP strategy thresholds:

| Field | Type | Default | Description |
|---|---|---|---|
| `ivrv_entry_threshold` | `float` | `1.15` | Min IV/RV ratio for entry |
| `iv_percentile_entry_min` | `float` | `30.0` | Min IV percentile for entry |
| `iv_percentile_entry_max` | `float` | `85.0` | Max IV percentile for entry |
| `max_vega_exposure` | `float` | `50000.0` | Portfolio vega limit |
| `max_delta_abs` | `float` | `0.3` | Portfolio absolute delta limit |
| `butterfly_wing_width` | `int` | `2` | Wing offset in strike gaps |
| `max_gamma_exposure` | `float` | `10000.0` | Portfolio gamma limit |
| `delta_hedge_threshold` | `float` | `0.5` | Delta threshold for hedge trigger |
| `profit_target_pct` | `float` | `0.50` | Exit at 50% of entry premium |
| `stop_loss_pct` | `float` | `2.0` | Exit at 2x premium loss |

**ModelConfig** - sklearn model configuration:

| Field | Type | Default | Description |
|---|---|---|---|
| `features` | `list[str]` | `[ivrv_ratio, iv_percentile, rv_30d, flat_forward_ratio, days_to_expiry]` | Feature columns |
| `target` | `str` | `forward_straddle_pnl` | Target variable |
| `test_size` | `float` | `0.2` | Test split ratio |
| `walk_forward_window` | `int` | `60` | Min training window for walk-forward |
| `model_type` | `str` | `regression` | `regression` or `classification` |

### Data Layer (`data/`)

**chain.py** - Option chain fetching and parsing from Groww API.

- `OptionContract` - Dataclass: `trading_symbol`, `strike`, `option_type`, `ltp`, `iv`, `delta`, `gamma`, `theta`, `vega`, `oi`, `volume`
- `OptionChain` - Dataclass with helpers: `atm_strike()`, `atm_contracts()`, `atm_iv()`, `straddle_premium()`, `get_contract(strike, type)`
- `fetch_option_chain(groww, underlying, expiry)` - Fetches and parses full chain from Groww API
- `fetch_expiries(groww, underlying, year, month?)` - Lists available expiry dates

**iv_engine.py** - Black-Scholes pricing and implied volatility.

- `bs_call_price()` / `bs_put_price()` - European option pricing
- `bs_vega()` - Vega for Newton-Raphson step
- `implied_vol()` - Newton-Raphson IV solver with Brenner-Subrahmanyam initial guess
- `realized_vol(prices, window=30)` - Annualized RV from log returns: `rolling_std * sqrt(252)`
- `compute_historical_iv_series()` - IV time series from daily option closes

**fno_historical.py** - 7-day paginated F&O candle fetcher.

- `fetch_fno_candles(groww, trading_symbol, start, end, interval=1)` - Paginates in 7-day chunks (Groww API limit), returns OHLCV DataFrame

**vol_surface.py** - Volatility surface analytics.

- `vol_cone(iv_series, percentiles=[10,25,50,75,90])` - Rolling percentile bands
- `term_structure(chains)` - ATM IV across expiries sorted by DTE
- `flat_forward_ratio(near_chain, far_chain)` - Near/far ATM IV ratio

**universe.py** - F&O liquidity screener.

- `screen_universe(groww, underlyings, min_volume, min_oi)` - Filters underlyings where ATM CE+PE both exceed volume and OI thresholds

### Signals (`signals/`)

**base.py** - Core signal types.

- `OptionType` - Enum: `CE`, `PE`
- `StrategyAction` - Enum: `SELL`, `BUY`
- `OptionLeg` - Frozen dataclass: `trading_symbol`, `strike`, `option_type`, `action`, `lots`
- `OptionsSignal` - Frozen dataclass: `underlying`, `expiry`, `legs`, `strategy_name`, `signal_strength`, `entry_iv`, `entry_ivrv`, `model_score`

**ivrv.py** - IV/RV ratio computation.

- `compute_ivrv(atm_iv, underlying_prices, rv_window=30)` - Scalar ratio; > 1 means IV overstates RV
- `compute_ivrv_series(iv_series, underlying_prices, rv_window)` - Vectorized time series

**iv_percentile.py** - IV percentile rank.

- `iv_percentile(current_iv, iv_history)` - Current IV rank vs history, 0-100
- `rolling_iv_percentile(iv_series, lookback=252)` - Rolling rank over lookback window

**flat_forward.py** - Term structure signal.

- `flat_forward_signal(near_chain, far_chain)` - Near/far ATM IV ratio; > 1 suggests near-term vol premium

**composite.py** - Composite signal snapshot and VRP entry logic.

- `VolSignalSnapshot` - Dataclass aggregating: `atm_iv`, `rv_30d`, `ivrv_ratio`, `iv_percentile_rank`, `flat_forward_ratio`, `days_to_expiry`
- `compute_snapshot(chain, underlying_prices, iv_history, far_chain?, rv_window)` - Builds snapshot from chain + historical data
- `is_vrp_entry(snapshot, config)` - True if IVRV >= threshold AND IV percentile in [min, max]

### Strategies (`strategies/`)

**options_base.py** - Abstract base class. All strategies implement:

| Method | Signature | Purpose |
|---|---|---|
| `generate_signal` | `(chain, underlying_prices, vol_snapshot, model_prediction?) -> OptionsSignal \| None` | Produce entry signal |
| `construct_legs` | `(chain, lot_size) -> tuple[OptionLeg, ...]` | Build option legs |
| `exit_condition` | `(chain, entry_signal, days_in_trade, current_pnl, vrp_config) -> bool` | Check if position should close |

**vrp_straddle.py** - Short ATM straddle. Sells ATM CE + ATM PE. Registered as `vrp_straddle`.

**vrp_butterfly.py** - Iron butterfly with defined risk. Sells ATM CE + ATM PE, buys OTM wings at `butterfly_wing_width * strike_gap`. Registered as `vrp_butterfly`.

Both strategies share the same exit logic:
1. DTE <= `days_before_expiry_exit` -> exit
2. PnL > `profit_target_pct` * entry premium -> exit
3. PnL < -`stop_loss_pct` * entry premium -> exit

**Adding a new strategy:**

1. Create `strategies/my_strategy.py`, subclass `OptionsStrategy`
2. Implement `generate_signal()`, `construct_legs()`, `exit_condition()`
3. Decorate with `@register_options` (from `strategies.registry`)
4. Import in `runner/options_cli.py` backtest and live commands so the decorator runs

### Models (`models/`)

**features.py** - Feature matrix construction.

- `build_feature_matrix(snapshots, config)` - Extracts configured feature columns from `VolSignalSnapshot` list into DataFrame
- `build_target(straddle_pnls, model_type)` - Regression: raw PnL. Classification: sign(PnL)

**pipeline.py** - Model training and persistence.

- `train_model(features, target, model_type)` - GradientBoosting with StandardScaler, 5-fold TimeSeriesSplit CV. Returns `ModelResult` with model, metrics (r2/mse or accuracy/f1), and predictions
- `walk_forward_backtest(features, target, window=60)` - Expanding-window walk-forward evaluation
- `save_model(result, path)` / `load_model(path)` - joblib persistence

**predict.py** - Live inference.

- `extract_features_from_snapshot(snapshot, config)` - Single-row DataFrame from snapshot
- `predict_vrp(snapshot, model_path, config)` - Load model + predict from one snapshot

### Backtesting (`backtesting/`)

**options_engine.py** - Expiry-cycle backtest engine.

The backtest iterates over expiry cycles within `[start, end]`:

1. Collect all expiries for the date range via `fetch_expiries()`
2. For each expiry, compute `entry_date = expiry - days_before_expiry_entry`
3. Fetch option chain at entry, compute `VolSignalSnapshot`
4. Call `strategy.generate_signal()` -- skip cycle if None
5. Record entry prices from chain LTPs
6. Walk forward day-by-day, re-fetching chains
7. Check `strategy.exit_condition()` each day; exit early on profit/stop-loss/DTE
8. If no early exit, close at `exit_deadline = expiry - days_before_expiry_exit`

Aggregation: total PnL, win rate, avg PnL, max drawdown (from cumulative PnL peak), Sharpe (annualized by trade frequency).

Lot sizes: NIFTY=25, BANKNIFTY=15, FINNIFTY=25, MIDCPNIFTY=50 (default=25).

**options_pnl.py** - Multi-leg P&L.

- `leg_pnl(action, entry, exit, lots, lot_size)` - BUY: `(exit-entry)*qty`, SELL: `(entry-exit)*qty`
- `trade_pnl(legs, entry_prices, exit_prices, lot_size)` - Sum across all legs

**options_report.py** - Visualization.

- `print_options_stats(result)` - Summary table (trades, total PnL, win rate, avg PnL, max DD, Sharpe)
- `plot_options_equity(result)` - Cumulative PnL equity curve
- `plot_vol_cone(iv_series, rv_series)` - IV vs RV time series with spread fill

### Execution (`execution/`)

**options_broker.py** - Per-leg FNO order placement.

- `OptionsBroker.place_legs(signal, lot_size)` - Places each leg as a separate market order via Groww API. Detects partial fills and logs error for manual intervention. Supports dry-run mode.

**options_safeguards.py** - Greeks-based risk limits.

`OptionsSafeguardChecker.check(signal, chain, capital, lot_size)` validates:
- Trading hours (IST window from `SafeguardConfig`)
- Daily loss limit (cumulative PnL / capital)
- Portfolio vega <= `max_vega_exposure`
- Portfolio gamma <= `max_gamma_exposure`
- Portfolio delta <= `max_delta_abs`

Also tracks portfolio greeks via `update_greeks()` and daily PnL via `on_trade_closed()`.

**delta_hedger.py** - Periodic delta neutralization.

- `DeltaHedger.compute_portfolio_delta(chain, positions, lot_size)` - Net delta across positions
- `DeltaHedger.check_and_hedge(chain, positions, lot_size)` - If `|net_delta| > delta_hedge_threshold`, places a futures hedge order to neutralize. Supports dry-run.

### Live Trading (`runner/`)

**live_vrp.py** - `LiveVRPRunner` polling loop.

Flow per poll cycle per underlying:
1. Fetch nearest future expiry
2. Fetch option chain
3. If position open -> check exit condition, run delta hedge check
4. If no position -> fetch 1Y historical, compute `VolSignalSnapshot`
5. Optional: `predict_vrp()` from saved model
6. `generate_signal()` from strategy
7. `OptionsSafeguardChecker.check()` -- block if limits exceeded
8. `OptionsBroker.place_legs()` -- execute
9. Track position in `_open_positions` dict
10. Sleep `poll_interval` (default 300s) between cycles

**Dry-run mode** (`--dry-run`, default): all orders are logged but not sent to Groww API.

## Entry Conditions

**Rule-based** (both must hold):
- `ivrv_ratio >= ivrv_entry_threshold` (default 1.15)
- `iv_percentile_entry_min <= iv_percentile <= iv_percentile_entry_max` (default 30-85)

**Optional ML filter**: if a model path is provided, `model_score > 0` is required.

## Exit Conditions

A position is closed when any of these trigger:
- **DTE exit**: `days_to_expiry <= days_before_expiry_exit` (default 1)
- **Profit target**: `current_pnl > profit_target_pct * entry_premium` (default 50%)
- **Stop loss**: `current_pnl < -stop_loss_pct * entry_premium` (default 2x)

## Configuration Reference

Configs are loaded as lazy `@property` on `Settings` (`config/settings.py`):

```python
settings.options  # -> OptionsConfig()
settings.vrp      # -> VRPConfig()
settings.model    # -> ModelConfig()
```

Each returns a dataclass with defaults. Override by mutating fields after access:

```python
settings.vrp.ivrv_entry_threshold = 1.20
settings.options.rv_window = 60
```

## Adding a New Strategy

1. **Create** `strategies/my_strategy.py`
2. **Subclass** `OptionsStrategy` and implement `generate_signal()`, `construct_legs()`, `exit_condition()`
3. **Decorate** with `@register_options` from `strategies.registry`
4. **Import** in `runner/options_cli.py` backtest and live commands (so the decorator registers the class)

```python
from strategies.options_base import OptionsStrategy
from strategies.registry import register_options

@register_options
class MyStrategy(OptionsStrategy):
    name = "my_strategy"
    # ... implement 3 abstract methods
```

Then run:
```bash
python main.py options backtest --strategy my_strategy --underlying NIFTY --start 2023-01-01 --end 2024-12-31
```
