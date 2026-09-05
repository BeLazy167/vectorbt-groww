"""Entry point for the vectorbt-groww trading pipeline."""

from dotenv import load_dotenv

load_dotenv()

from runner.cli import cli

if __name__ == "__main__":
    cli()
