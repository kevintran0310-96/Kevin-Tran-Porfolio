
import click
import pandas as pd
from pathlib import Path
import math

@click.group()
def cli():
    pass

@cli.command()
@click.option("--base-path", default="data/raw/samples")
def ingest(base_path):
    path = Path(base_path) / "transactions_sample.csv"
    df = pd.read_csv(path)
    print(f"Ingested {len(df)} rows from {path}")

@cli.command()
@click.option("--base-path", default="data/raw/samples")
def transform(base_path):
    path = Path(base_path) / "transactions_sample.csv"
    df = pd.read_csv(path)
    df["amt_log"] = (df["amt"]+1).apply(math.log)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/transactions_transformed.csv", index=False)
    print("Transformed dataset saved.")

@cli.command()
@click.option("--base-path", default="data/raw/samples")
@click.option("--out-dir", default="data/processed")
@click.option("--fig-dir", default="reports/figures")
def queries(base_path, out_dir, fig_dir):
    import matplotlib.pyplot as plt
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(fig_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(base_path)/"transactions_sample.csv")
    fraud_rate = df["is_fraud"].astype(int).mean()
    print(f"Fraud rate: {fraud_rate:.2%}")

    plt.hist(df["amt"], bins=30)
    plt.title("Transaction Amounts")
    plt.savefig(Path(fig_dir)/"amt_hist.png")

if __name__ == "__main__":
    cli()
