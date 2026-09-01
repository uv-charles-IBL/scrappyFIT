# scrappyFIT

Light-element PIXE spectrum fitting, mapping and quantification.

An independent Python implementation of the physics in CSIRO's **GeoPIXE**,
extended where the light-element regime (C at 277 eV through F at 677 eV)
needed it. It reads GeoPIXE's own file formats, so the two can be run side by
side and compared directly.

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

```bash
pip install -e .
python -m scrappyfit
```

Needs GeoPIXE's `database/` directory. Point at it with **File → Set database
folder**, the `SCRAPPYFIT_DB` environment variable, or by copying it into
`scrappyfit/resources/database` so it travels with the code.

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
tools/        database rebuild and validation scripts
docs/         user manual
```

## Documentation

[docs/user_manual.md](docs/user_manual.md) — workflow, the physics, and how to
read a result critically.

[CHANGELOG.md](CHANGELOG.md) — what changed and whether it moves numbers.

## Status

0.1.0, early. The physics is validated against samples of known composition;
the GUI is new.
