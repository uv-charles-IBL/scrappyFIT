scrappyFIT — Python companion to CSIRO's IDL GeoPIXE for light-element PIXE.

**Local paths.** This file is the same on every machine. Where things live on this one (the
lab knowledge base, the GeoPIXE source, the lab spectra, data roots) is in `CLAUDE.local.md`
at the repository root. It is untracked (`.gitignore`), and git worktrees do not get a copy:
read it from the main checkout, the parent of `git rev-parse --git-common-dir`. If there is
no `CLAUDE.local.md`, none of the lab resources below are available. Say so, and work from
the repository alone.

Before working here read `docs/user_manual.md` (maintained; authoritative on workflow), then
in the lab knowledge base: `README.md`, `methods/pixe-fitting-scrappyfit.md`,
`methods/geopixe-internals.md`, and `projects/light-element-pixe/README.md`.
Check the knowledge base's `decisions/RETRACTED.md` before repeating any striking claim —
several conclusions about this code and about GeoPIXE have been retracted.

Write new findings into the knowledge base's `projects/light-element-pixe/FINDINGS.md` in
the same turn, and update `methods/pixe-fitting-scrappyfit.md` plus the `pixe-fitting` skill
when a lesson changes how the task is done. Agent memory holds index cards only.

Every number carries its script and its date.
GeoPIXE licence circumvention is off the table; do not inspect the licence verifier.

## Tests

The suite must be green. `python -m pytest -q` runs everything (`testpaths = tests`).
- Fast suite: `python -m pytest -q -m "not slow"` (~10 s). The pre-commit hook runs it.
  Enable the hook once per clone: `git config core.hooksPath .githooks`.
- Golden spectra: `tests/test_golden_spectra.py`, marked `slow` (~1-2 min). Five lab
  reference spectra fitted with their analysis recipes, results pinned. They lock in current
  behaviour, not truth. Re-pin a value only with a CHANGELOG line saying what moved it and
  why; never loosen a tolerance to get green.
  **The pins are machine-specific.** They reproduce only on the lab desktop where they were
  made. Other machines land three fits in a different minimum with the same numpy and scipy,
  so CI skips them. The lab spectra are not public: they live untracked in `tests/data/` or
  under `SCRAPPYFIT_LAB_SPECTRA` (`tests/labdata.py`; a worktree also finds the main
  checkout's `tests/data/`), and without them every golden case skips. With
  `SCRAPPYFIT_REQUIRE_LAB_DATA=1` a missing spectrum fails instead; set it for the
  pre-merge run. On the lab desktop, run them before every merge. An outside contributor cannot run
  them and does not need to: the maintainer runs them before merging.
- External data: `SCRAPPYFIT_GEOPIXE_DATA` lists the directories (os.pathsep-separated) that
  hold GeoPIXE's example files. A few tests and most `tools/` scripts read from them, and
  the tests skip without them. Without the lab spectra, the GUI journeys and the smoke fit
  run on `tests/data/synthetic_quartz.txt` (`tools/make_synthetic_spectrum.py`).
- CI: `.github/workflows/tests.yml` runs on every push: the fast suite on Linux under
  Python 3.10 (pinned `ci-constraints.txt`), 3.11 and 3.12, and the slow GUI journeys on
  Windows (budgets x2). The golden spectra are skipped there.

## Roles

**Maintainer** (changes code):
- Works on a branch `fix/<slug>`, one BUGS.md item per session.
- Failing test first: the `test-writer` agent reproduces the BUGS.md line as an
  `xfail(strict=True)` test; the fix removes the marker.
- Runs the `reviewer` agent on the diff before committing, and addresses its blocking findings.
- Asks the `geopixe-ref` agent, not memory, what GeoPIXE's IDL does.
- Ticks the BUGS.md line with the fixing commit, adds a CHANGELOG `Unreleased` line
  (physics that changes fitted numbers is always called out).
- Never merges to master and never force-pushes. Charles merges.

**Analyst** (uses scrappyFIT on data):
- Never edits code in this repo. Pins a release tag (`git checkout vX.Y.Z` or
  `pip install git+...@vX.Y.Z`) and records the tag beside every result.
- Files problems as one line in `BUGS.md` in its format (repro, expected vs actual, date).
  Physics findings go to the lab knowledge base, not BUGS.md.

**Releases**: tagged `vX.Y.Z` on master by Charles, each with a CHANGELOG entry and the
matching `version` in `pyproject.toml`.

## Agents (`.claude/agents/`)

| agent | model | does |
|---|---|---|
| `reviewer` | inherit | read-only review of a diff: units, physics, GeoPIXE conventions, tests, RETRACTED.md |
| `test-writer` | sonnet | one failing (xfail-strict) test reproducing one BUGS.md line; edits `tests/` only |
| `geopixe-ref` | sonnet | what GeoPIXE's IDL does, with `file.pro:line` citations from the GeoPIXE source (`CLAUDE.local.md`); read-only |
