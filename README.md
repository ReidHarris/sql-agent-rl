# sql-agent-rl

A training pipeline for a text-to-SQL agent.

## Installation

Requires Python 3.10+ and \[conda](https://docs.conda.io/en/latest/miniconda.html).

```bash
git clone https://github.com/ReidHarris/sql-agent-rl.git
cd sql-agent-rl
conda create -n sqlagent python=3.11
conda activate sqlagent
pip install -e ".\\\\\\\[dev]"
```

## Running the Tests

```bash
pytest
```

## Data

1. Download the [Spider dataset](https://yale-lily.github.io/spider)
2. Extract so that `data/spider/dev.json` and `data/spider/database/` exist.

