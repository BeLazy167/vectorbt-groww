# vectorbt-groww

Python research prototype for Indian-stock strategy backtesting with Groww market data and vectorbt. Includes performance reports, options strategies, and experimental trading execution code.

## Capabilities

- Stock backtesting with moving-average crossover, Bollinger-band mean reversion, opening-range breakout with RSI, and multi-confluence strategies.
- Equity curves and trade reports.
- Options volatility calculations, short-straddle and iron-butterfly strategies, and model-training code.
- Experimental paper and live execution through Groww.

## Status

This project is unfinished. The options backtest fetches current option quotes while iterating historical dates, so it does not provide a valid historical simulation. The stock live command also uses symbol placeholders for exchange tokens. These paths need work before use. Dependencies are not pinned. The CLI help command has been checked, but backtests and trading execution have not been validated end to end.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set your Groww credentials in `.env`. Keep this file local.

Display the available commands:

```sh
python main.py --help
python main.py backtest --help
python main.py options --help
```

See [OPTIONS.md](OPTIONS.md) for the options implementation notes. The limitations above take precedence over its backtesting examples.

## Local files

Credentials, personal financial reports, generated charts, data caches, and Python environments are excluded from Git.

## License

The source code in this repository is available under the [MIT license](LICENSE). Dependencies retain their own licenses.
