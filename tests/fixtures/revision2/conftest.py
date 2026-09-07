"""Exclude revision2 fixtures from pytest auto-collection.

These test files ship as evidence tied to the 6636670 fix (buffered auth
admission retention + Responses JSON scalar validation). They:

- Live under ``tests/fixtures/revision2/`` (a fixtures path, not a suite path).
- Depend on standalone unittest execution (``python -m unittest ...``),
  not pytest's session-wide state.
- Install a global ``logging.getLogger().handlers=[Capture()]`` at import time
  which clobbers pytest's caplog / assertLogs isolation for every later test.

Skipping them here keeps the suite green under ``pytest tests/``; run them
explicitly via ``python -m unittest discover tests/fixtures/revision2`` when
you need the raw revision2 forensic evidence.
"""

collect_ignore_glob = ["test_*.py"]
