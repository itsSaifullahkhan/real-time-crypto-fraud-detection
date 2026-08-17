# Data

This folder holds local-only project data when a generator phase explicitly creates it.

- `raw/`: future raw source extracts or downloaded market files
- `generated/`: local generated customer, event, market-candle, audit, and label datasets
- `sample/`: future small safe samples for documentation and tests

Phase 5A writes the local pilot to:

```text
data/generated/historical/pilot/
```

Pilot data is for generator validation only. It is not uploaded to ADLS and must not be committed.
