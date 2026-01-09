# vr-foraging-analysis

## Requirements

- UV to manage reproducible environments (https://docs.astral.sh/)
- awscli to sync data from S3 (https://aws.amazon.com/cli/) (optional, only if you want to sync data locally)

## Getting started

1. Bootstrap the environment:

   ```bash
   uv sync
   ```

2. Run the `demo.ipynb` notebook to see example analyses.

## Syncing data locally

A dataset is fully defined in a yml file called `sessions.yaml`. Which is picked up automatically by the code by running:

```python
settings = DataLoadingSettings()
```

From here, you can sync the dataset locally by running:

```python
from ssvr.s3_utils import sync_dataset

sync_dataset(settings)
```

Note: if some of the assets are in private buckets, make sure you [AWS credentials are properly configured](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html).
