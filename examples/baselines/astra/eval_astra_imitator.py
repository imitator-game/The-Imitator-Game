"""Verify or run the recorded Astra workflow; paid execution requires --execute."""
from .astra.runtime import main

if __name__ == '__main__':
    raise SystemExit(main())
