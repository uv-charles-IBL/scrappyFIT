# Changelog

Versions follow [semantic versioning](https://semver.org). Physics changes
that alter fitted numbers are always called out, because a result is only
reproducible against a stated version.

## Unreleased (since v0.27.0)

## 0.27.0 - 2026-10-21: fits independent of BLAS threads; Maps tab with GeoPIXE image tools

### Physics that changes fitted numbers
- Fits no longer depend on the BLAS thread count (BUGS.md "Fitted numbers depend on the BLAS
  thread count"). The linear step's Gram product, matrix-vector products and inverse no longer
  use BLAS (einsum, and an equilibrated Gauss-Jordan inverse with one refinement step, in
  `fitting/fit.py`). The prior-scale first pass is the same normal-equations solve, not
  `lstsq`. `least_squares` runs with every BLAS held to one thread (threadpoolctl). That hold
  is process-wide: while it lasts, BLAS calls from other threads in the process also run on
  one thread. Results are bit-identical at `OMP_NUM_THREADS` 1, 4 and 16 (quartz, 287424) and
  across numpy 1.26 / 2.5 for 287424 (to 1e-14). The solve's residual is 7e-16 on 287424's
  cond-2e15 system, against 4e-12 for `lstsq` (scratch `accuracy.py`, 2026-10-21). The
  desktop's old 287424 value (chi2r 2.3259) came from its 16 BLAS threads. Master at one
  thread gives 2.3816, the value Linux and the GitHub runner gave. 287424 now takes 3.5x longer on the lab desktop (152 s vs 43 s, same
  load) because its optimiser path is longer, not because each step costs more (BUGS.md,
  2026-10-21; scratch `count.py`, 3 interleaved runs, 20:32-20:43). Quartz is about 20 %
  faster (7.4-7.8 s vs 8.9-12.0 s, same runs). Concurrent fits share the one-thread hold
  (reference-counted). A missing threadpoolctl is an error, not a silently skipped refinement.
  Golden values moved (`tests/test_golden_spectra.py`; values from its `run_case` on the lab
  desktop, scratch `case.py`, 2026-10-21; quartz and 404001 stay within tolerance and are
  unchanged; the fit sigmas of the re-pinned areas were re-pinned with them):
  - 287424: chi2r 2.32594 -> 2.38158, noise 34.7563 -> 35.9506, Fano 28.1844 -> 26.911, Si
    216083.7 -> 217090, IL 34682.27 -> 34459.95, PbM5 31030.22 -> 31083.23, AgL 27167.23 ->
    26946.47, Ca 18128.72 -> 17966.84, PbM4 15722.73 -> 15724.84, InL 15007.76 -> 14742.33, BrL
    13308.95 -> 14115.8, C 9918.004 -> 9828.795, IM3 8629.967 -> 8644.389, PbMz 6297.208 ->
    6299.316, IM2 6238.232 -> 6092.393, Mg 4636.299 -> 4544.464, Na 4551.021 -> 4896.446, SnL
    4133.331 -> 4310.125, K 3210.854 -> 3307.243, CsM1 3209.907 -> 2691.509, PbM3 2332.423 ->
    2379.517, Al 2159.324 -> 1432.126 (now 2.7 sigma, unpinned), Cl 372.0896 -> 318.4642.
  - 404003: chi2r 1.34791 -> 1.35018, noise 38.1076 -> 38.1036, Fano 24.991 -> 25.0451, cal a[2]
    1201.58 -> 1201.62, AlKb 1399442 -> 1363270, Al 554646.8 -> 555028.3, GaL 125174.9 ->
    125181.3, O 65828.76 -> 65828.28, AuM5 39061.07 -> 39112.67, AuM4 22251.34 -> 22203.59, Ti
    7152.967 -> 7153.517, AuM3 3764.213 -> 3764.421, AuMz 2281.801 -> 2307.723, N 1533.301 ->
    1537.792, AuM2 997.8632 -> 999.222, Si 728.5026 -> 720.6023.
  - 404004: Fano 21.9016 -> 21.9466.

### Dependencies
- New runtime dependency: threadpoolctl >= 3.0 (BSD-3-Clause; CI pins 3.6.0 in
  `ci-constraints.txt`). It holds BLAS to one thread around `least_squares`.

### Maps (map numbers change; no fitted number changes)
- Smoothed window maps no longer read low on the outer pixels. `analysis/maps.py` `smooth` zero-padded
  the field, so a flat map read 30 % low on the edge rows, 51 % in the corners and 6 % one pixel in;
  it now divides by the kernel weight inside the field. On 287424 (Si, raw window, bin 4) the edge
  rows went from 64 % to 92 % of the interior mean and the corners from 41 % to 85 % (Ag L: 64 -> 92 %,
  51 -> 104 %) (`D:\Claude gens misc\maps-phase3\edge_check.py`, 2026-10-21). Interior pixels are
  unchanged. Near the edge this moves window-map values, CSV exports, Profiles traverses on window
  maps, the default colour limits, and the pixels a percentile mask picks (so the spectrum and fit of
  a region masked from a map that touches the edge). Edge pixels now average fewer neighbours and are
  noisier. DA maps and unmasked fits are unaffected.
- Net window maps take their background flanks only from channels that hold events: from the zero of
  energy (and the spectrum's first count) to its last count. A flank clipped to under half the peak
  window's width is dropped and the background is flat from the other flank; with neither, the map
  refuses ("map it without subtraction"). `Session.map_info` records the window, the flanks and which
  background was used. On 287424 the first count is channel 305 (0.146 keV), so C (flank 159-249) and
  N (229-318) lose their low flank: mean net per bin-4 pixel C 1.05 -> -3.64 and N 8.14 -> 6.43,
  O and Si unchanged (`D:\Claude gens misc\maps-phase3\flank_check.py`, 2026-10-21). The C map
  now goes negative because its only flank, 518-608, sits on O K: a window map cannot separate them;
  use a DA map. A line near the top of the spectrum likewise no longer reads an empty high flank.
  A percentile mask cut from a net C or N map now selects different pixels, so that region's
  spectrum and fit change; unmasked fits do not.
- Masks work at a binning that does not divide 256 (3, 5, 6, 7, ...). Window and DA maps are now
  ceil(256 / binning) pixels a side (binning 3: 86, was 85), so the last raster row and column have a
  pixel: `nbin // binning` wrapped x = 255 into the next row and dropped the top row, and a rectangle
  on the map raised `MASK FAILED` (IndexError). The last pixel is partial; a DA map divides each pixel
  by the charge of its own raster pixels, so it reads like the rest; a window (counts) map's last row
  and column read low in proportion (a third at binning 3), so a percentile mask skips them. The mask
  overlay sits on the raster's true extent. `analysis/masking.py` gains `bin_factor`, `to_raster` and
  `map_fraction`; the Maps tab passes the binning with every mask, because the map size alone is
  ambiguous above binning 15 (16 and 17 both make 16 px) and `Session.set_mask` now refuses an
  ambiguous map-shaped mask unless `binning=` says which.
  At binnings that divide 256 (1, 2, 4, 8, 16, 32; default 4) no map, mask or DA number changes; a
  script passing a 16- or 8-px mask to `set_mask` must now add `binning=16` or `binning=32`.

### Maps tab: map kinds and backgrounds
- Every map says what it is. A **window** map (counts in an energy window) or a **DA matrix** map
  (wt% through a loaded GeoPIXE .dam) is chosen in the first box, and the title (`WINDOW, net counts`,
  `WINDOW, raw counts`, `DA, wt%`), the colour-bar units and a line under the controls (keV window,
  channels, background and flanks, or the matrix and its row) state it. Loading a DA matrix switches
  the tab to DA maps of the matrix's elements; before, DA maps reached only Concentration profiles.
- Window background choices as GeoPIXE's cut types (`analyze_cut.pro`; geopixe-ref 2026-10-21):
  linear between two flanks (default, as before), flat from the low or the high flank, or none. The
  "subtract background" checkbox is gone: its default (ticked) is the combo's default, linear, and
  unticked is "none". `Session.element_map(background=...)`, `Session.da_map`,
  `Session.da_map_names`. No map number changes for the default (linear) or for "none".
- **table...** beside the line list opens a periodic table of the map lines (window lines in the fit
  range, or the DA matrix's elements) in the element dialog's layout; each cell shows the shells it
  can map, one click maps a single shell and a cell with several asks which (`Fe` / `Fe L`). GeoPIXE
  picks from a droplist only (`gimage.pro:1636-1647`). `gui/maplines.py`.
- **Associations...** plots one map against another as GeoPIXE's Element Associations (`corr.pro`;
  geopixe-ref 2026-10-21): a 2-D histogram of the two maps' pixel values (made as the Maps tab makes
  them: same kind, background and binning), density in log colour, optional log axes 5 decades deep,
  negatives clipped to 0, range 1.05 x the maximum inside a 2 % border, and Pearson r. A polygon drawn
  round a cluster selects every pixel whose point lies inside (GeoPIXE tests the histogram bin; here
  the pixel's own point, so the selection does not depend on the bins) and replaces, adds to, refines
  (AND) or takes from the mask, after GeoPIXE's include / refine / reject (whether its reject takes
  from the current selection or the field was not checked). Every pixel is on the plot: a hot pixel
  in the border sits on the top edge. Pearson r is of the values as plotted (negatives at 0). Window maps are not divided by
  a charge map as GeoPIXE's axes are (no charge map here yet). `analysis/association.py`,
  `gui/association.py`.

## 0.26.0 - 2026-10-21: spectrum display, Sabbatucci-Salvat fetched on first use

### Fixes
- The error dialog is not shown on Qt's offscreen platform (headless tests, CI): `QMessageBox.show()`
  crashed the interpreter there (PyQt5 5.15.11 / Qt 5.15.2), so any GUI test that ran before an
  excepthook test took the whole pytest run down. The error is still logged; the log also records
  "error dialog not shown". Real desktops are unchanged.
- Licence: `MAC_SabbatucciSalvat2016.txt` is no longer in the repository or the wheel. It is
  converted from EGSnrc's AGPL-3.0 `photo_shellwise.data`, which cannot ship inside an MIT
  package; v0.25.0 and earlier shipped it, and there it is under AGPL-3.0 (THIRD_PARTY_NOTICES.md).
  `python -m scrappyfit.io.sabbatucci` (or `fetch()`) downloads EGSnrc's file at a pinned commit
  and rebuilds the table, byte-identical to v0.25.0's (SHA-256 checked), into `~/.scrappyfit/data`;
  choosing it in the GUI's MAC box offers that download behind a consent dialog. Selecting
  `sabbatucci` without it now raises `SabbatucciMissing` saying how to get it. No fitted number
  changes: `mixed` (the default) never used it, and the fetched table is the same file.

### Tests and process
- GUI journey timing checks allow a fixed 200 ms on top of budget x scale (`JITTER_MS`,
  `tests/test_gui_journeys.py`). Sub-200 ms steps were failing on disk and scheduler noise
  (J3.export_yield on CI, J5.open1 on the desktop). No budget value changed; no fitted numbers change.

### Spectrum display
Display only: no fitted number moves (golden spectra bit-identical, fix/spectrum-display).
- Spectrum legend beside the plot, never over the data: as many columns as it needs, the plot
  narrows by its width, and past 40 % of the width the last element entries fold into
  "+ n more". **Lines...** picks the element groups drawn and listed, kept across fits
  (BUGS.md "Choose which lines are drawn and listed").
- **escape + pile-up curves**: the fit's escape lines and pile-up terms drawn as two summed
  curves, as GeoPIXE's Escapes / Pileup overlays (which GeoPIXE builds from the data).
  `FitResult.components` and `escape_spectrum()`, `Component.escape_profile()`.
- Si escape markers move down 0.3 eV onto the database Si Ka_ (1.740 keV, escape_energy.pro:21)
  where the fitted escape lines already were; the fitted escape lines are unchanged.
- Ge detectors: K-beta escape markers at E - Ge Kb1 (branching 0.133), as GeoPIXE draws.
- No more "wrapped C/C++ object ... has been deleted" from a draw queued as a window closed.

## 0.25.0 - 2026-10-21: GUI rework, maps, legend, layered yields; first public release

The public repository starts from a snapshot of this state (2026-10-21). Commit hashes and CI
run numbers below, and in every older entry, refer to the private archive of the earlier
history (versions 0.1.0 to 0.24.0).

### Physics that changes fitted numbers
- GUI: a fit no longer starts from the last fit's shape. Unless "Hold line shape from last fit"
  is ticked, every FIT refines from the detector width and the calibration boxes, so a fit with
  the wrong elements cannot spoil the next (BUGS.md "A misfit's line shape poisons every later
  fit"). 287424.lmf, published recipe, fit / drop Al Si C / add back: before 2.382, 359.373,
  8.411 (Ag L 7051); after 2.382, 48.147, 2.382 with the first fit's shape and Ag L 26946
  (headless MainWindow repro, 2026-10-21). Refits after a change cost a free fit again
  (J2.refit_noF about 7 s, was about 2 s areas-only). Ticked with no fit yet now really refines
  freely, as its message said (before, the options shape was frozen). New "Reset line shape"
  button; opening a file or changing ADC forgets the shape. GeoPIXE seeds each fit from its
  detector too (`pixe_setup.pro:106-107`) but writes a free fit's width back for the next
  (`pixe_cleanup.pro:75-80`); scrappyFIT deliberately does not. The live estimate draws the
  shape the next fit starts from. An unheld region refit now also starts from the detector shape,
  not the last field fit (J6.refit_region about 11 s, was about 2 s), so region fits that should
  share the field's shape need "Hold line shape" ticked, as the manual says.
- `Session.run_fit` with `start_pars` set and `freeze_nonlinear` off now refines from that start
  on the first pass; before, `fit_spectrum` took the start as converged and solved the areas
  only (refining once more only in the pile-up pass). Scripts that set `start_pars` get
  different numbers. With `start_pars` None (the golden recipes) the path is unchanged.
- A carried line shape (`Session.start_pars`; in the GUI the held one) is moved to the current
  range before it is used: noise^2 += fano^2 x the change in E low, and the eoc channel moves by
  a[3] (channels per keV) x the change in eoc, so a held fit after an E low or E high edit keeps every line's width and
  position (`Session._start_shape`, used by `run_fit` and `preview`). Before, the old noise was
  read at the new E low and the old eoc channel at the new eoc. The live estimate with a held
  shape now uses the held calibration, as the held fit does, instead of the calibration boxes.
  Golden recipes unchanged (no start_pars).
- Changing E low keeps the detector width: `FitOptions.e_low` re-references `noise`
  (noise^2 += fano^2 x the change in E low), so the FWHM at every energy stays put, as GeoPIXE's
  `pixe_setup.pro` does by rebuilding a0 from w0/w1 at the current E low. Before, a `.detector`,
  `estimate_resolution` or hand width kept the number referenced to the old E low. Fits that set
  the range after the width now start from a different width. The five golden recipes are
  unchanged (104 passed, Charles's desktop, 2026-10-19). On 404004 with free calibration that
  start choice can flip the result between basins (chi2r 3.83 old vs 3.74 new, detector-then-range
  order; range-then-detector 2.12; `tools/width_elow_order.py`, 2026-10-19). `tools/compare_to_geopixe.py`
  now sets the .pfr's E low before its width.
- GUI: E low / E high take effect when edited, not at FIT. A `.detector` loaded after typing a
  range now converts its w0/w1 at the typed E low, not the stale one. Run 404004 in the J3
  journey's order (open, cal, range 0.17-6.0, detector, deflector efficiency, C N O Al Si, FIT):
  the start noise moves 30.906 -> 30.645 ch, and the fit converges to a different solution,
  chi2r 3.737 -> 2.171 (`tools/gui_fit_dump.py`, 2026-10-18, 5fb8fed vs
  this branch). The library is unchanged: old and new give bit-identical results on all five
  golden recipes, and the old GUI with the range applied first gives the new numbers exactly.
  Kept deliberately (Charles, 2026-10-18): the old order fitted with a width referenced to an
  E low that was no longer in force. Correction (2026-10-19): the size of that split is not the effect of
  the 0.8 % width. Start noise values one ulp apart give chi2r 3.736 and 2.120 on the same recipe
  (`tools/width_elow_order.py`; D:\lab-knowledge RETRACTED R19).
- L-shell line list follows `list_line_index.pro`: only L-alpha is lumped below the split; L-beta,
  L-gamma, Leta and Ll are always individual. Run 287424: chi2r 2.958 -> 2.382, In L x2.54, Sn L
  x0.66, Al x0.36.
- QUANTIFY reads the sample's GeoPIXE .yield files when it has them; this changes QUANTIFIED
  numbers, not fitted ones. It now defaults to the yield set whenever one is found or chosen: one
  file per UNKNOWN layer, and per region for files named `<stem>_<region>_UNKNOWN-<layer>.yield`.
  The set is remembered per data file, found beside the data, or on a search path
  (`~/.scrappyfit/yields.json`, `SCRAPPYFIT_YIELD_STORE`). Each file gives only the elements of
  its UNKNOWN layer, in ppm of that layer, as GeoPIXE uses one column per file
  (`pixe_fit.pro:414`). Before, QUANTIFY bootstrapped one K-shell layer and could not report Ag,
  I, Pb, In or Sn. Run 287424 (plateau / void / bare, charge 1.507 uC x pixel fraction, 2.2 msr)
  reproduces the lab's published per-region ppm (`yield_v2_gp.json`, 2026-10-12; Pb M fitted as subshells, the file's M5+M4 share) to 1.2e-4 for Ag,
  I, Pb, In, Sn, Si, Na, Ca and Mg, from the published areas and from a refit
  (`gui_yieldset_check.py`, 2026-10-20). Br and Al differ because the publication split the
  1.48 keV peak by hand. The output names every file used, or says the matrix is bootstrapped
  and why. Fitting is untouched. An exported .yield does not switch QUANTIFY to it
  (`Session.quantify_yields`, `scrappyfit/yieldset.py`, `tests/test_yield_set.py`).

### GUI: observable, scriptable, responsive
- Action log: every handler's name, arguments, duration and context (file, cal, range,
  elements) as JSON lines in `~/.scrappyfit/logs/<date>.log` (`SCRAPPYFIT_LOG_DIR`).
  `sys.excepthook`, `threading.excepthook` and a Qt message handler log errors there and show
  a non-modal dialog instead of aborting (`gui/observe.py`).
- Fits, batches and the live estimate run on one worker thread (`gui/worker.py`) with
  progress, Cancel and a `finished(kind, result)` signal; both `processEvents()` calls are gone.
  Controls that change the session are disabled while a fit or batch runs.
- Live estimate, as GeoPIXE's fit setup draws it: `Session.preview(elements)` gives peak-height
  areas (`pixe_initial.pro:117-230`: lines above 0.01 of the major line, height = window sum
  minus background over 2j+1, area = height x 1.0645 x FWHM / rel. int., min over those > 0.1,
  floor 3) and one model evaluation, no pile-up. Redrawn on element, cal and range changes,
  coalesced over 150 ms. Its SNIP curve is cached on the window, never handed to a fit.
- All drawing goes through `sessionChanged` / `previewUpdated` -> `refresh_spectrum`.
- Cooperative cancel in the library: `fit_spectrum(should_stop=)`, `Session.run_fit(should_stop=,
  progress=)`, `run_batch(should_stop=)`; `FitCancelled`. Defaults leave results bit-identical.
- The session owns the sample's `.yield` (`Session.yield_path`): the file File > Export yield
  wrote, one loaded in any dialog, or one found beside the data (`<stem>.yield`, or the single
  `.yield` whose `.provenance.json` names the data file; GeoPIXE has no such convention, only the
  `.pcm` yield path, `fit_setup.pro:2597-2611`). Layer stack and escape opens on it and says which
  file; Sample model shows it, with Change... and Copy its composition. QUANTIFY never reads the
  file, so fitted and quantified numbers are unchanged (goldens 5/5 and journeys green, Charles's
  desktop, 2026-10-20; `tests/test_gui_yield.py`, 12f5755).

### Fixes
- Layer stack and escape no longer crashes on a multi-layer .yield (`Y.get('thick') or [...]`
  on a numpy array). Every multi-layer file failed, 287424's among them. It now opens on the
  region's file with the UNKNOWN layer selected, and lists the rest of the set. (`tests/test_yield_set.py::test_gui_quantify_reads_the_yield_set_and_says_so` crashed before.)
- `PixeDB.line_energy(Z, shell)` / `major_line` honour the shell: L is L-alpha (`La_` up to Z 52,
  `La1` above), M is `Ma1`; K unchanged bit for bit. Before, every shell returned the K line (Ag L
  at 22.16 keV). Callers checked: spectrum line labels (`MainWindow.line_labels`: L and M labels
  now drawn at their lines, were off the plot), artefact markers (`artefacts.predict`, from the GUI
  only: L and M parents now get escape and sum predictions), Concentration profiles (`on_profiles`:
  L and M elements now mapped, were skipped as out of range), `layerview` line pick,
  `tools/compare_nigel_354016.py` peak labels, `Session.map_line_energy`. No fitting or yield path
  passes shell 2 or 3 (`unexplained_peaks` reads L/M from `lineE` itself; yields and
  efficiencies use the K default).
- Maps: the element box lists lines in the fit range (`Si`, `Ag L`, `Pb M`) and
  `Session.element_map` maps that shell (`parse_line`, `map_line_energy`: K major, L-alpha,
  M-alpha1). Before, `Ag` meant Ag K at 22 keV, so Ag, In, Sn, I, Cs and Pb mapped as all-zero maps
  on 287424 (0.2-6.5 keV); "Mask above percentile" on one selected 65536 of 65536 pixels as
  "top10%", and every later fit was a region fit (Charles's action log, 2026-10-20 22:13). A line
  off the spectrum now raises, and a percentile mask on a flat map is refused. Stepping through all
  115 entries on 287424 leaves the spectrum and fit untouched (headless MainWindow, 2026-10-21;
  `tests/test_maps_lines.py` locks it in on synthetic events); the
  refit change Charles saw after browsing was the carried misfit shape (cbf04fb). Bare symbols and
  energies still work as before.
- Maps: one colour bar serves every Show map instead of a new one each time. Under matplotlib
  3.11 each new colour bar chained the previous one through `cax.remove`, keeping every old map
  and its data alive; later scale, limit and colour-map changes are wired to each new map so the
  bar follows them. Display only.
- Maps: Show map no longer shrinks and moves the map on every click. The colour bar has its
  own axes made once (`MapCanvas.cax`) instead of `fig.colorbar(ax=...)` taking room each call;
  it sits 10 px right of the map as drawn, with 90 px kept for it at any window width.
- Maps: grow / shrink act on the current mask whichever tool made it (rectangle and percentile
  masks were ignored) and step one displayed pixel (the binning, 4 raster pixels by default,
  not 1), with a square kernel so a rectangle stays a rectangle. The active mask is now drawn
  on the map (tint and outline). `masking.grow` / `shrink` take `connectivity=8`; the default
  4 is unchanged.
- Maps: the map has a zoom / pan toolbar, and **Reset view** returns to the whole field and
  the default contrast (2nd-99th percentile), keeping the mask. Flood-fill clicks are ignored
  while zooming or panning. Reset sets a new norm rather than `set_clim` (which, from a narrow
  contrast, came back at the wrong minimum).
- GUI left panel no longer clipped: it is never narrower than its content (horizontal scrollbar
  off), the calibration formula moved out of the group box title, the database path wraps at
  its separators, the spectrum toolbar is two rows, and the panel's notes use pt sizes so they
  scale with Windows DPI. Real window at 1500x950, 2026-10-20: before, the window was forced to
  1589 / 1747 / 1938 px at font x1 / 1.25 / 1.5 and the panel cut off; after, 1500 px at all
  three with nothing cut off (`tests/test_gui_layout.py`). Display only.
- GUI Artefacts button no longer crashes: `artefacts.predict` takes `escape_fraction` and the markers
  use the fit's own `EscapeModel`. Display only; fitted numbers do not change.
- Artefact escape markers follow the detector crystal (`options.crystal_Z`), as GeoPIXE's
  `escape_energy.pro`: Ge escapes at E - Ge Ka1 (db) above the Ge K edge (db); crystals other than
  Si and Ge show none. Si markers are bit-identical to before. Display only; fitted numbers do not
  change (escape lines in the fit come from `EscapeModel`, untouched).
- DA matrix (`io/gpda_write.build_matrix`) solves on unit-norm rows: a 1e9-norm background row
  had zeroed every DA weight (blank E+ maps on run 404003).
- `.yield` writer follows GeoPIXE's reading: atom fractions in F, 1-based UNKNOWN, major line first
  with relative intensities, Y before detector efficiency.

### Added
- Spectrum legend box (beside log y / components / line labels): compact (data, fit,
  background, pile-up and escape; default), all (every drawn element component too), off.
  Display only; fitted numbers unchanged (golden spectra 5/5, Charles's desktop, 2026-10-20).
- Region legend: with a mask active the data entry names the region, its raster pixels and
  counts, and the **whole field** checkbox draws the whole-field spectrum behind it, scaled by
  the region's share of the scanned pixels (equal counts per pixel, not equal charge).
- `load_detector` fills area and distance from the `.detector` file (never over a typed value);
  GUI 'Load .detector file...' button.
- `FitOptions.sum_deficit_spread`: sum peaks as a band of deficits for flat-topped pile-up humps.
- GUI 'Hold line shape from last fit' (freeze_nonlinear) and a progress message for long fits.
- `LayeredYieldModel.yields(e_lines=...)`.
- Maps colour scale: colour table (magma, viridis, inferno, cividis, gray, turbo), linear / log
  (floor 5 decades below the top, as GeoPIXE's G, `build_image_scale.pro:39-42`) / sqrt (of the
  value above the low limit), and low / high limits as percentiles of
  the map (GeoPIXE uses % of maximum, which one hot pixel sets). Settings carry over to the next
  Show map; Reset view restores the 2-99 percentile limits. `MapCanvas.set_display`.
- Maps: previous / next arrows beside the element list map the neighbouring element in one click
  (wrapping); GeoPIXE has the droplist only (`gimage.pro:1636-1647`).
- Maps: **Export map...** writes the shown map as CSV (row 0 = bottom, header with file, element,
  net/raw, binning, calibration) and, if a mask is set, `<name>_mask.csv` with the fraction of each
  displayed pixel inside it. `gui.app.export_map_csv`.
- Maps: polygon mask tool (click corners, click the first to close), and a mask op - replace /
  add / subtract - so shapes build a region as GeoPIXE's include / exclude do
  (`gimage.pro:1676-1679`). Grow / shrink ignore the op; subtract with no mask cuts the shape
  out of the whole field. The rectangle and flood-fill seed now
  take the pixel under the cursor (pixel i spans i-0.5..i+0.5); before, a drag starting in the
  left or lower half of a pixel began one pixel early.

### Tests and process
- GUI journey J6 (maps, on `287424.lmf`; skipped where the file is absent, e.g. CI): whole-field
  fit, Show map twice (the map must not move), top-decile mask, grow, shrink, zoom + Reset view,
  refit of the region. Budgets from `tests/gui_timings_maps.json`. `tests/test_gui_maps.py`: fast
  MapCanvas and Maps-tab tests (21).
- Golden regression tests (`tests/test_golden_spectra.py`, marked `slow`): runs 404001 MnOx,
  404003 Ga2O3/Au/Ti, 404004 SiC, 287424 perovskite and the quartz, fitted with their analysis
  recipes; chi2r, calibration and the main areas pinned 2026-10-18 from 6c0718b. Characterization,
  not truth. Spectra in `tests/data/` with the C1 deflector `.detector` file.
- `pyproject.toml` sets `testpaths = ["tests"]` and the `slow` marker; bare `pytest` no longer
  collects `tools/`.
- `.githooks/pre-commit` runs the fast suite; `.github/workflows/tests.yml` runs everything on push.
- `.claude/agents/`: `reviewer`, `test-writer`, `geopixe-ref`.
- Golden spectra are a desktop-only check: CI skips them (`CI=true`; `SCRAPPYFIT_RUN_GOLDEN=1`
  forces them). Their pins reproduce only on the machine they were made on. 287424 gives
  chi2r 2.3816 against the pinned 2.3259 on Linux and on a GitHub Windows runner with the same
  numpy 1.26.4 / scipy 1.11.3 (CI run 39296724668, 2026-10-19). Pins and tolerances unchanged.
  CI installs `ci-constraints.txt`, runs the fast suite on ubuntu-24.04 and the slow GUI
  journeys on windows-2022 with `SCRAPPYFIT_BUDGET_SCALE=2`.
- GUI journeys J1-J5 (`tests/test_gui_journeys.py`, pytest-qt, offscreen, marked `gui` and
  `slow`) with per-step budgets in `tests/gui_budgets.json` (1.5x measured; before/after timings
  beside it). `tests/test_gui_infra.py` covers cancel, the preview rule, the action log and the
  job runner.

### Speed (fitted numbers bit-identical)
- Fits 1.3-3.7x faster with every number unchanged: chi2, ndf, areas, errors, all shape parameters
  and the full model, background and profile arrays are bit-identical to master 7f873fb on all five
  golden recipes (`tools/golden_dump.py`, Charles's desktop, 2026-10-20), for finite parent
  areas (a NaN or inf pair weight now poisons only its own window of the plateau). Wall time per recipe,
  7f873fb -> this branch: quartz 7.5 -> 4.3 s, 287424 23.4 -> 17.7 s, 404001 39.9 -> 10.8 s,
  404003 42.0 -> 16.5 s, 404004 3.1 -> 2.2 s. How: each distinct sum-peak line is profiled once
  ((i, j) and (j, i) are the same line); the pile-up plateau and the empirical-response (E+,
  `hypermet`) tail and shelf skip erfc where it is exactly 0 or 2 (`peakshape.erfc_saturated`); `de_shelf` forms its S-independent
  terms once; the low-statistics filter's scalar maths runs on Python floats.
- `tools/bench_fit.py` times a fixed numpy workload and the quartz fit with the machine's state
  (power plan, CPU clock, BLAS threads), to tell a slow machine from slow code.
  `tools/golden_dump.py` dumps the five golden fits bit-exactly and compares two dumps.

### Docs
- Manual: fit timing and held-shape region fits; detector-file geometry; known limits. NNLS note in
  `fit_spectrum` corrected. Known open issues are in `BUGS.md`.

### Packaging, licences and portability
- Licences: `LICENSE` (MIT, Copyright (c) 2026 University of North Texas). CSIRO's GeoPIXE MIT
  licence ships beside the eleven database files copied unmodified from it
  (`resources/database/dat/LICENSE-CSIRO-GeoPIXE`), and `THIRD_PARTY_NOTICES.md` gives the source
  and licence of every bundled data file. `MAC_SabbatucciSalvat2016.txt` is converted from EGSnrc
  data, which EGSnrc distributes under AGPL-3.0-or-later; whether it stays is Charles's decision,
  open. The `Database.mu` docstring no longer says 'mixed' uses Sabbatucci above the edge (it uses
  XCOM). No numbers change.
- The lab's five reference spectra are no longer tracked. They stay untracked in `tests/data/` on
  the lab desktop (a worktree also finds the main checkout's), or under `SCRAPPYFIT_LAB_SPECTRA`
  (`tests/labdata.py`), and the tests load the same files there as before.
  `SCRAPPYFIT_REQUIRE_LAB_DATA=1` makes a missing spectrum fail instead of skip. Without them the
  golden spectra skip, and the GUI journeys, the smoke fit and the E-low width test run on
  `tests/data/synthetic_quartz.txt`, made by `tools/make_synthetic_spectrum.py` from scrappyFIT's
  own model (no lab data; refits to chi2r 1.011, 2026-10-20). CI's Windows journeys therefore time
  the synthetic spectrum against budgets measured on the lab spectra. The lab files stay in the
  git history until Charles decides on a rewrite. The `.detector` geometry tests and J6 are
  unaffected.
- `physics/kl_agree.py` no longer holds a hard-coded database path. The constant was unused.
- Tests that need GeoPIXE's example files (yields, `.dai`, run log session) look for them under
  `SCRAPPYFIT_GEOPIXE_DATA`, a list of directories separated by `os.pathsep`, and skip without it.
  The `.detector` geometry tests use the identical copy in `tests/data/` and now run everywhere.
- `tools/` scripts take their input paths as arguments, defaulting to files under
  `SCRAPPYFIT_GEOPIXE_DATA` (`tools/_paths.py`). `make_ffast.py` and `rebuild_lines.py` default to
  reading the vendored database and write to the current directory. `make_ffast.py` reproduces
  `MAC_FFAST.txt` byte for byte; `rebuild_lines.py` does not reproduce `xray_lines_rebuilt.txt`
  (BUGS.md, 2026-10-20).
- Version 0.25.0.dev0 (was 0.7.0), kept in one place (`scrappyfit.__version__`, read by
  `pyproject.toml`); set it to the release number in the tagged commit. `scrappyfit-cli` is now a
  headless batch fit (`scrappyfit/cli.py`) instead of a second GUI launcher. Python 3.10+.
  `requirements.txt` is gone: `pyproject.toml` is the dependency list. The wheel carries `LICENSE`
  and `THIRD_PARTY_NOTICES.md`. The CI fast job runs on 3.10 (pinned `ci-constraints.txt`), 3.11
  and 3.12 (latest releases).
- README: installing from a tag or a clone, the database, running the window, the CLI and the
  tests, and the licence.
- `CLAUDE.md` and `.claude/agents/` name no local paths. They read `CLAUDE.local.md` (untracked)
  for the knowledge base and GeoPIXE source, and work from the repo alone when it is absent.

## 0.9.0 to 0.24.0 (from the release tags)

These releases were tagged without CHANGELOG entries. The list below is the tag
messages, added 2026-10-20; `git log vA..vB` has the detail.

- 0.24.0 (2026-09-29): M lines for Z = 45-72, frozen region fits, escape-shelf truncation fixed
- 0.23.2 (2026-09-24): fast-threshold-aware pile-up; sloped-shelf option for measured responses
- 0.23.1 (2026-09-24): sum peaks at the line width, refinable sum deficit, partial pile-up plateau
- 0.23.0 (2026-09-24): measured detector response; refinement fixed to run after pile-up and to
  score by the reported chi2
- 0.22.3 (2026-09-13): sub-threshold pile-up component for spectra taken with the fast channel off
- 0.22.2 (2026-09-13): Amptek DPPMCA .mca reader with DPP-setting warnings; Session.load accepts .mca
- 0.22.1 (2026-09-13): .yield layer fractions are ATOM fractions; thick-target yields now agree
  with GeoPIXE
- 0.22.0 (2026-09-13): SNIP follows GeoPIXE's defaults; trim_seb ported; the 354016 over-read
  explained
- 0.21.1 (2026-09-12): GeoPIXE .dai reader, and the topaz DA matrix was never degenerate
- 0.21.0 (2026-09-12): 380006 was missing arsenic, not an M4/M5 problem; width bounded;
  unexplained-peak report
- 0.20.0 (2026-09-11): He-PIXE fit corrected: background, M subshells
- 0.19.0 (2026-09-11): .2D reader; Ga2O3 comparison
- 0.18.0 (2026-09-11): Levenberg-Marquardt refinement
- 0.17.0 (2026-09-11): helium beams, seeded resolution, Ga2O3 comparison
- 0.16.0 (2026-09-04): sum-peak weighting and pile-up display
- 0.15.0 (2026-09-02): efficiency-weighted line intensities
- 0.14.0 (2026-09-02): PIXE quantification validated above Z=20
- 0.13.1 (2026-09-02): validation against the worked example
- 0.13.0 (2026-09-02): energy-dependent detector tail
- 0.12.0 (2026-09-02): GeoPIXE detector reader; example comparison agrees
- 0.11.0 (2026-09-01): GeoPIXE pile-up and escape models
- 0.10.0 (2026-09-01): profiles, .spec reader, GeoPIXE comparison
- 0.9.0 (2026-09-01): GeoPIXE fit-result reader; public DA loading
- 0.8.1 to 0.8.5 (2026-09-01): run log reader and digitiser quantum; absolute scale traced and
  fixed; absolute quantification closes; Z trend traced to line grouping; offset narrowed,
  calibration verified, GUI command line

## 0.8.0 - artefacts in the fit, and absolute quantification

### Physics that changes fitted numbers
- **Silicon escape peaks are now fitted components**, tied to their parent
  line rather than free. The escape fraction follows `escape_fraction.pro`,
  but with the prefactor that GeoPIXE keeps in `detector.GAMMA` written out
  explicitly: `omega_K * (1 - 1/r) / 2`, taken from the database so a
  germanium crystal gets germanium's numbers. For silicon this is 0.0229.
  Treating `GAMMA` as 1, which is the obvious reading of the IDL, gives a
  calcium escape fraction of 42% instead of 0.96%.
- **Pile-up is a fitted component** with a free amplitude and a shape that is
  the self-convolution of everything else, so it follows the other components
  instead of being a fixed correction applied afterwards.
- Escape peaks are correctly absent below the Si K edge at 1.839 keV, so no
  light-element line generates one.

### Absolute quantification
- `physics/geometry.py`: solid angle, dead-time live fraction, and the
  `calc_yield.pro` normalisation constant, so a result can be reported in
  weight percent without normalising to 100.
- `Session.quantify(absolute=True)` and `Session.closure()`, the diagnostic
  that a normalised total cannot provide.
- GUI: a "Charge and solid angle" panel. Filled in, QUANTIFY reports absolute
  concentrations and judges the total; left blank, it normalises as before
  and says so.

### File formats
- `lmf.clocks()` returns the per-block dose counter and the run duration.
  Field 0 is digitiser pulses, not microcoulomb - the conversion is not in
  the file. Fields 1-4 are four microsecond clocks that track each other to
  within ~500 us, so **dead time cannot be recovered from an LMF** and the
  function returns `live_fraction = None` rather than a fabricated 1.0.

### Fixed
- The artefact button now reports the escape and pile-up terms the fit
  actually used, instead of an independent prediction that could disagree
  with the curve on screen.


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
