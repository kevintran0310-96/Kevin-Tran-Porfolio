import json
from click.testing import CliRunner
from portfolio_etl.cli import cli
from pathlib import Path

def test_ingest_on_sample(tmp_path, monkeypatch):
    # point to bundled sample
    base_path = Path('data/raw/samples')
    assert (base_path / 'transactions_sample.csv').exists()

    runner = CliRunner()
    result = runner.invoke(cli, ['ingest', '--base-path', str(base_path)])
    assert result.exit_code == 0
    assert "Ingested" in result.output

def test_transform_and_queries(tmp_path):
    base_path = Path('data/raw/samples')
    out_dir = Path('data/processed')
    fig_dir = Path('reports/figures')

    runner = CliRunner()
    r1 = runner.invoke(cli, ['transform', '--base-path', str(base_path)])
    assert r1.exit_code == 0
    assert (out_dir / 'transactions_transformed.csv').exists()

    r2 = runner.invoke(cli, ['queries', '--base-path', str(base_path), '--out-dir', str(out_dir), '--fig-dir', str(fig_dir)])
    assert r2.exit_code == 0
    # histogram should be created
    assert (fig_dir / 'amt_hist.png').exists()