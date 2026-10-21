# scrappyFIT

Light-element PIXE spectrum fitting, mapping and quantification.

An independent Python implementation of the physics in CSIRO's **GeoPIXE**,
extended where the light-element regime (C at 277 eV through F at 677 eV)
needed it. It reads GeoPIXE's own file formats, so the two can be run side by
side and compared directly.

> **History.** This public repository starts from a single snapshot of 0.25.0.dev0
> (21 October 2026). Development before that, versions 0.1.0 to 0.24.0 and their tags,
> is kept in a private archive. Commit hashes and CI run numbers cited in `BUGS.md` and
> `CHANGELOG.md` from before that date refer to the archive.

## Why

Four specific things went wrong when pushing GeoPIXE below 1 keV, each of
which produced a real error in a real measurement:

- **the shipped line database contains fabricated values** — Ca, Sc, Ti and V
  share one placeholder L-line pattern, and Ca and Sc are given 89% Lα, a
  transition they cannot make
- **most attenuation databases stop at 1 keV**, which is above every light
  element of interest
- **the detector response model is incomplete below ~600 eV**, and the missing
  few percent gets assigned to whatever element has a line nearby
- **nothing checks whether a fitted element is physically possible**

Together those produced a chondrite spectrum reporting **13,425 counts of
titanium** in a sample whose own Ti K line accounts for 725 — and the same
artefact, identically, in a quartz containing no titanium at all.

## Install

You need Python 3.10 or newer (3.10, 3.11 and 3.12 are tested) and Git.

Install a tagged release straight from GitHub, replacing `vX.Y.Z` with a tag
from the [tags list](https://github.com/uv-charles-IBL/scrappyFIT/tags). This
install works from v0.25.0 on; earlier tags predate the packaging fixes. Pin
the tag and record it beside your results, so a number can be traced to the
code that made it:

```bash
pip install "scrappyfit[dev] @ git+https://github.com/uv-charles-IBL/scrappyFIT@vX.Y.Z"
```

Or clone the repository and install that checkout:

```bash
git clone https://github.com/uv-charles-IBL/scrappyFIT.git
cd scrappyFIT
pip install ".[dev]"
```

Use `pip install -e ".[dev]"` if you are going to change the code. `[dev]` adds
pytest and pytest-qt, which you only need to run the tests. A virtual
environment (`python -m venv .venv`) keeps the dependencies (numpy, scipy,
matplotlib, PyQt5, xraylib, xraydb) apart from your other projects.

**Atomic database.** The package ships the atomic data it needs in
`scrappyfit/resources/database`, so it works without GeoPIXE installed. To use
a GeoPIXE `database/` directory instead, choose **File → Set database folder**
in the window or set the `SCRAPPYFIT_DB` environment variable.

**Run it.**

```bash
scrappyfit                      # the window (or: python -m scrappyfit)
scrappyfit spectrum.txt --cal 0.0016984,-0.37225 --elements C,N,O,Si --fit
scrappyfit-cli run1.lmf run2.lmf --elements C,N,O,Si --out results/
```

`scrappyfit-cli` fits files without opening a window. Every file gets the
same settings, and the per-run exports plus `batch_summary.csv` are written
to `--out`.

**Run the tests.** From a clone:

```bash
python -m pytest -q -m "not slow"
python -m pytest -q -m slow
```

The first command is the fast suite, about 15 s. The second runs the full fits
and the GUI journeys, a few minutes. The lab's reference spectra are not
public, so on your machine the tests that need them either skip (the golden
spectra) or run on `tests/data/synthetic_quartz.txt`, a spectrum made by
`tools/make_synthetic_spectrum.py` from scrappyFIT's own model. If you have
the spectra, or GeoPIXE's example data, point the tests at them with
`SCRAPPYFIT_LAB_SPECTRA` and `SCRAPPYFIT_GEOPIXE_DATA` (see `tests/labdata.py`).
The golden values are pinned on the machine that made them. On other hardware
three of the five fits land in a different nearby minimum, so treat a golden
failure on your machine as a difference of hardware, not a bug.

## What it does

- reads OMDAQ list-mode (`.lmf`), GeoPIXE `.dam` and `.yield`, text spectra
- fits with a detector response that includes Si LVV Auger escape and window
  contact injection, which GeoPIXE does not model
- selectable mass attenuation datasets including a `mixed` rule measured to be
  the most accurate for silicates
- L and M line data rebuilt from xraylib
- elemental maps, and **region masking** — mask a phase on the map and refit
  just that region
- physical screening: Coster–Kronig, M/L corroboration, K-vs-L agreement
- exports the spectrum, model, every component, residual, areas, mask and
  **every option used** into one self-describing folder

## Layout

```
scrappyfit/
  io/         file readers
  physics/    atomic data, fundamental parameters, yields
  fitting/    peak shape, response, background, least squares
  analysis/   maps, masking, database benchmarking
  gui/        the application
  session.py  the facade the GUI and scripts both drive
tools/        database rebuild, benchmark and validation scripts
tests/        pytest suite (tests/labdata.py: where external data is found)
docs/         user manual
```

## Documentation

[docs/user_manual.md](docs/user_manual.md) — workflow, the physics, and how to
read a result critically.

[CHANGELOG.md](CHANGELOG.md) — what changed and whether it moves numbers.

## Status

0.25 in development (last release v0.24.0). The physics is validated against samples of known composition, and
the GUI is still young. See the CHANGELOG for what moved in each release.

## Licence

scrappyFIT is released under the MIT licence, Copyright (c) 2026 University of
North Texas (see [LICENSE](LICENSE)).

The atomic data it bundles keeps its own terms, listed file by file in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Most of it comes from CSIRO's
open-source GeoPIXE (MIT, Copyright (c) 2023 CSIRO), whose licence text ships
beside the data. The rest is NIST FFAST, CXRO/LBNL Henke data, a table from
EGSnrc (AGPL-3.0-or-later; see the notices), and tables derived for scrappyFIT.
GeoPIXE is CSIRO's software and trade mark. scrappyFIT is an independent
implementation and is not endorsed by CSIRO.
