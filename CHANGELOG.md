# Changelog

Versions follow [semantic versioning](https://semver.org). Physics changes
that alter fitted numbers are always called out, because a result is only
reproducible against a stated version.

## 0.1.0 - first cut

Extracted from a working session against GeoPIXE and reorganised into a
package. Everything below already existed as scripts; this release is the
point at which it became maintainable.

### File formats
- OMDAQ list-mode (`.lmf`) reader, validated to one event in 206,934 against
  a known PIXE spectrum
- GeoPIXE Dynamic Analysis matrix (`.dam`) reader, including the DA projection
  so concentration maps can be produced the way GeoPIXE produces them
- GeoPIXE `.yield` layer-model reader
- `hubbell.dat` reader (the Berger-Hubbell / XCOM tables)

### Physics
- Mass attenuation datasets selectable at run time: Henke 1993, Sabbatucci-
  Salvat 2016, FFAST (Chantler), XCOM, and a `mixed` rule
- `mixed` follows Heirwegh 2014: FFAST below each absorber's K edge and
  everywhere below 1 keV, XCOM above. Below 1 keV no XCOM-class tabulation
  exists at all, so this is not a preference but a necessity for light
  elements. Scores best on a quartz sample of exactly known composition.
- L and M line data rebuilt from xraylib (Campbell & Wang rates). The shipped
  GeoPIXE table gives Ca, Sc, Ti and V an identical fabricated 100:10:1:1
  pattern, and gives Ca and Sc 89% La - a transition they cannot make, having
  an empty 3d shell.
- Coster-Kronig factors taken from xraylib, not Elam: Elam reports non-zero
  f23 for the 3d metals, but that channel is energetically forbidden below
  about Z=30.
- Mn Lb/La corrected from 0.301 to 0.203, interpolated across its neighbours.

### Detector response
- Si LVV Auger escape step, extent max(90 eV, 8% of the line energy)
- Contact-injection step from the window metallisation, edge at the Al L3
  binding energy
- Both use the sloped difference-of-exponentials form of Heirwegh 2014 rather
  than a flat step
- Si3N4 window term implemented; measured amplitude is zero on this detector

Without these terms a fit assigns the unexplained intensity to whatever
element has a line nearby. On a chondrite that produced 13,425 counts of
"titanium" in a sample whose own Ti K line says 725.

### Screening
- Coster-Kronig floor test: rejects an L assignment whose L3 population is
  below what f13 and f23 force
- M/L corroboration: rejects an M assignment inconsistent with the same
  element's L intensity
- K-vs-L agreement: both shells must give the same concentration

### Known limits
- The binary GeoPIXE `.spec` format is not read yet, only the text export
- Shelf slopes are fixed; fitting them per energy needs monochromatic spectra
- No OMDAQ live connection

## 0.2.0 - efficiency, batch, masks, interop, live

- **Detector efficiency curves** (`physics/efficiency.py`). Reads GeoPIXE EFF3
  exports in both layouts; three curves vendored. `from_layers()` computes one
  from an absorber stack for what-if questions. `Session.quantify()` turns
  fitted areas into wt%, validated on quartz: O 53.79 against a true 53.26.
- **Batch processing** (`batch.py`). One option set across many files, a
  per-run export plus one wide `batch_summary.csv`. Failures are recorded and
  the batch continues.
- **Mask tools** (`analysis/masking.py`). Flood fill with a relative
  tolerance, polygon, grow, shrink, boolean combination. Flood tested on the
  fluorine inclusions in 287427: 15x F and 7x Al enrichment over the field.
- **DA matrix writing** (`io/gpda_write.py`). scrappyFIT results can now go
  back into GeoPIXE. Round-trips to 100.51 wt% on quartz against quantify()'s
  own answer. Note the charge convention: GeoPIXE applies a matrix as
  `matrix . spectrum / charge`, so the stored weights must carry a factor of
  charge or every concentration is silently out by the run's charge.
- **Live list-mode reading** (`io/live.py`). An LMF is a header plus complete
  8192-byte blocks, so a reader can consume whole blocks as they land with no
  locking and no cooperation from OMDAQ. Verified against a simulated growing
  file: 599,294 events, identical to a static read.
- GUI: efficiency selector, QUANTIFY, wt% column, Batch dialog, DA export,
  live attach with a 1 s poll, flood/grow/shrink mask tools.

## 0.3.0 - self-contained, and it can tell you what is in the spectrum

- **Database vendored** into `scrappyfit/resources/database` (5.9 MB, 15
  files). A fresh install now needs nothing else - it runs off a flash drive.
  An external database still wins if one is configured.
- **Line identification** (`physics/lineid.py`). Click the spectrum for
  ranked candidate lines at that energy.
- **Element suggestion** from the whole spectrum. Peaks are found against a
  local background, then elements scored on whether their lines explain them.
  Three rules reject coincidences: the strongest visible line must be
  present, conspicuously absent lines are penalised, and matches are weighted
  by peak size. Verified on four samples - quartz gives O/Si, the perovskite
  gives Si/Pb/I/In, salt rock gives Na/Cl, gold-on-carbon gives C/AuM.
  Coincidences still appear lower down; it is a shortlist, not a verdict.
- GUI: identify-on-click, Find peaks, Suggest from spectrum with highlighting
  in the element lists, Accept suggested.

## 0.4.0 - every element reachable, peaks marked and labelled

- **Element lists are now built from the energy range**, not hard-coded. At
  0.2-6.5 keV that is 21 K, 51 L and 45 M entries; widen the range and the
  transition-metal K lines appear. A fixed menu is worse than useless on an
  unknown sample - it silently rules out whatever the author did not think of.
  A filter box narrows the list by symbol.
- **Identification markers.** Clicking the spectrum now draws a dashed guide
  where each candidate line WOULD fall under the current calibration. If the
  guide misses the peak, either the identification or the calibration is
  wrong, and both are worth knowing before fitting.
- **Label all peaks** annotates every detected peak with its best
  identification, preferring elements already in the fit. Teal means the
  element is being fitted, red means the line table suggested it and nothing
  else supports it.
- **Save spectrum image** (Ctrl+P) writes exactly the view on screen at
  200 dpi - markers, zoom and all.
- Labels are placed into whichever of five rows is free at that energy and
  dropped rather than overprinted, so a crowded low-energy region stays
  readable. The x-axis is now clamped to the fit range; previously a guide
  for a line at 30 keV would stretch the axis and squeeze the data into a
  sliver.
