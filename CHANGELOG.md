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

## 0.5.0 - periodic table, sample model, yield writer

- **Periodic table element selector**, replacing the fixed lists. Each cell
  cycles none - K - L - M - KL - LM - KLM, skipping any shell with no line in
  the window, so one repeated click walks the possibilities. A second tab is a
  read-only line viewer: click an element, read its table. Browsing what an
  element emits and deciding to fit it are different acts, and a viewer that
  silently changed the fit would be a trap.
- **Sample model dialog** (`Analysis - Sample model`). Matrix either
  bootstrapped from the fitted areas or typed in, plus thickness, beam energy
  and take-off angle. Every concentration depends on these and they were
  previously invisible. After quantifying, the result is offered back as a
  typed matrix so refining is one click.
- **Yield file writer** (`io/gpyield_write.py`). Version -3 deliberately: from
  -8 the format embeds an IDL beam struct whose byte layout would have to be
  guessed, and a calibration file that loads but is subtly wrong is the worst
  outcome. The MAC provenance field only exists from -12, so the real settings
  go in a sidecar .provenance.json rather than being silently lost.
- Fixed: the x-axis was not clamped, so a guide for a line outside the window
  stretched the plot to 78 keV.

## 0.6.0 - pile-up, contrast, layer stack viewer

- **Sum-peak pile-up is now a fitted component.** Two photons inside the
  shaping time record as one at the sum energy. Small - 0.27% of counts on
  quartz - but it lands in otherwise empty regions, which is exactly where a
  fit invents an element. The unexplained 2.24 keV peak in the quartz
  spectrum was the Si Ka + O Ka sum, and was being offered to mercury and
  niobium. Confirmed three ways: energy 2.235 against 2.265 predicted, FWHM
  107 eV against 53 for a single line, and a Si+O to Si+Si ratio of 1.90
  against 1.67 predicted. Modelling it took chi2r from 2.105 to 1.925.
- **Higher contrast plotting.** Components now cycle through eight colours
  rather than sharing one, the model is drawn last so it is never buried, and
  line weights are heavier. With ten overlapping elements a single hue was an
  unreadable thicket.
- **Layer stack viewer** (Ctrl+L). Draws the stack on a log thickness scale -
  the perovskite cell spans 0.002 to 2200 um and is meaningless linearly -
  and computes what fraction of a chosen X-ray escapes from each layer. On
  that cell: Pb Ma escapes 99.97% from the absorber, O Ka escapes 0.086%.
  Reports the mean free path against the layer thickness and warns when only
  the top of a layer is being sampled.

## 0.7.0 - artefact prediction and layered quantification

- **Show escape + sum peaks** button. Marks where silicon escape peaks and
  pile-up sums fall, before any element is chosen. Both put real sharp
  features where no element emits, and a fit offered one reaches for the
  nearest element. Predictions on quartz match measurement to 1%: O+Si sum
  predicted 484 counts at 2.265 keV, measured 488 at 2.235.
  The pile-up fraction is INFERRED from a clean sum peak in the data rather
  than assumed, and reported as unavailable when no clean one exists.
- **Layered quantification.** `quantify_layered()` for known stack / unknown
  concentrations, `solve_thickness()` for known composition / unknown
  thickness, and `instrument_constant()` underneath both.
- **solve_thickness requires an internal standard**, because it must. The
  yield model returns arbitrary units; quantify() cancels that by normalising
  to 100 wt%, but an absolute thickness cannot be normalised away. One
  element of known concentration in a known layer supplies the constant.
- **Adaptive depth stepping.** A layer thinner than one integration step
  returned the same yield however thin it was - a 5 and a 50 ug/cm2 film gave
  identical answers and a thickness solve had nothing to bisect on. Step count
  now scales so the thinnest layer gets at least 25 steps.
- Validated on the salt rock carbon film: 5.42 ug/cm2 calibrated on chlorine,
  5.54 on sodium (2.1% apart), against 5.6 from the independent
  internal-standard route earlier in the project.
