# Outlook Cortex (EmailOps Edition)

> **CRITICAL:** This workspace operates in **Cloud-Only Mode**.
> *   **DO NOT** run `docker-compose up` for databases.
> *   All development connects to DigitalOcean Managed Infrastructure.
> *   Refer to `docs/CANONICAL_BLUEPRINT.md` for the authoritative architecture and rules.

## Setup

1.  **Install `doctl`** and authenticate.
    ```bash
    doctl auth init
    ```

2.  **Environment**
    Ensure your `.env` is configured for `OUTLOOKCORTEX_ENV=production` and points to the cloud resources.

3.  **Run Locally**
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/backend/src
    python -m cortex.main
    ```

## Documentation

*   [Canonical Blueprint](docs/CANONICAL_BLUEPRINT.md) - The single source of truth.
*   [Implementation Ledger](docs/IMPLEMENTATION_LEDGER.md) - History of changes.
