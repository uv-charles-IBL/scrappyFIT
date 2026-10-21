# scrappyFIT user manual

Version 0.3.0

Light-element PIXE spectrum fitting, mapping and quantification. Reads OMDAQ
list-mode files and GeoPIXE's own formats, so it can be used alongside GeoPIXE
and compared against it directly.

---

## 1. Why this exists

GeoPIXE is a mature and careful piece of work, and most of the physics here is
a port of it. scrappyFIT exists because the light-element regime — carbon at
277 eV through fluorine at 677 eV — exposed four specific limits, each of
which caused a real error in a real measurement.

**The database contains fabricated numbers.** `xray_lines.txt` gives calcium,
scandium, titanium and vanadium an identical `100:10:1:1` L-line intensity
pattern. It is a placeholder, not a measurement. Worse, it gives calcium and
scandium 89% of their L intensity in the Lα line — a transition they
physically cannot make, because Lα is L3→M4,5 and their 3d shell is empty.
Their entire L3 emission is Lℓ, at a different energy. The Lℓ intensity is
understated by a factor of 11 to 13 for Ti–Mn and about 112 for Ca and Sc, and
all of those lines land in the O–F region.

**Below 1 keV, most attenuation databases contain nothing.** XCOM and Scofield
simply stop at 1 keV. Every light element of interest is below that. Only
FFAST (Chantler) and Henke cover the range.

**The detector response model is incomplete below about 600 eV.** GeoPIXE
models a peak as a Gaussian plus one low-side exponential tail. Real silicon
detectors also produce an Auger-escape step and a contact-injection step,
together worth a few percent of each light peak. That missing intensity does
not vanish from the fit — it gets assigned to whichever element has a line
nearby.

**Nothing checks whether a fitted element is physically possible.** A fit will
happily report titanium that is not there.

The last two combined to produce the case that motivated this program: a
chondrite spectrum that reported **13,425 counts of titanium L emission** in a
sample whose own titanium K line accounts for 725. The same artefact appears,
identically, in a quartz sample containing 53 counts of titanium K — that is,
none.

---

## 2. Installing

Python 3.10 or newer. The README's Install section has the full steps.

### From a checkout

```bash
pip install -e ".[dev]"
```

### From a release tag

```bash
pip install "scrappyfit @ git+https://github.com/uv-charles-IBL/scrappyFIT@vX.Y.Z"
```

### Dependencies

`numpy`, `scipy`, `matplotlib`, `PyQt5`, `xraylib`, `xraydb`. All available
from PyPI and installed by pip.

### Without the window

`scrappyfit-cli FILE... --elements C,N,O,Si --out DIR [--cal g,o] [--range lo,hi]
[--efficiency PATH] [--quantify]` fits every file with the same settings and
writes the per-run exports and `batch_summary.csv` to `DIR` (`batch.run_batch`).

### The atomic database

**A copy is vendored in the package**, so a fresh install works with no
GeoPIXE present - including off a flash drive. Nothing needs configuring.

To point at a different one — a newer GeoPIXE tree, say — supply a directory
containing `dat/` with `ElamDB12.txt`, `xray_lines*.txt`, `MAC_*.txt`,
`xsect_K/L/M.txt` and `hubbell.dat`. Resolution order, first hit wins:

1. a path passed to `config.set_database_path()`
2. the `SCRAPPYFIT_DB` environment variable
3. a copy vendored at `scrappyfit/resources/database`
4. the path saved in `~/.scrappyfit/config.json`
5. a GeoPIXE tree sitting beside the checkout

In the GUI: **File → Set database folder**. It is remembered. The vendored
copy is used unless you override it.

### Running

```bash
python -m scrappyfit
```

**The action log.** Every button, menu item, file load and element change is
written to `~/.scrappyfit/logs/<date>.log`, one JSON line each. A line holds
the action's arguments, how long it took, and the state around it: file,
calibration, fit range and elements. Set `SCRAPPYFIT_LOG_DIR` to put the log
somewhere else. When something breaks, that file is the record of what was
done, in order. Send it with the bug report. Unexpected errors go to the same
file with their traceback and raise a warning box, and the session carries on.

---

## 3. The workflow

The basic sequence mirrors GeoPIXE's, with two additions that are the reason
this program is useful.

### 3.1 Open a file

**File → Open**. Four formats:

| format | what you get |
|---|---|
| `.lmf` | OMDAQ list mode. Spectrum **and event positions** — maps and masking become available |
| `.dam` | GeoPIXE DA matrix. Calibration, charge, the DA projection; pulls in the source `.lmf` if it can find it |
| `.spec` | GeoPIXE **text** export (SPEC/DATA records). The binary `.spec` is not read yet |
| `.txt` | two columns, channel and counts |

**Open the `.lmf` wherever you can.** Everything in section 3.5 depends on
having event positions, and a summed spectrum does not carry them.

The **ADC** selector picks the detector: 0 is normally the PIXE detector, 1 is
usually RBS, 2 STIM, 3 IBIC. This varies by run — check the header strings in
the Log tab.

### 3.2 Set the calibration

`E = offset + gain × channel`. Type the values and press Enter.

Then **Analysis → Check calibration against known lines**. This fits the
centroid of each selected element's main line and reports the error in eV.

This is not a formality. A 60 eV error moves oxygen Kα onto the artefact
discussed in section 5, and light-element lines are only 100–150 eV apart. On
a well-calibrated setup you should see errors within ±20 eV; if you see 50 eV
or more, fix it before believing anything else.

### 3.2b Identify what is in the spectrum

Two tools, and they answer different questions.

**Click on the spectrum** (with *identify on click* ticked) to list the X-ray
lines near that energy, ranked. This is what you reach for when the residual
shows a feature you did not model. It lists candidates - it does not decide,
because at any energy in a light-element spectrum there are several honest
answers.

**Suggest from spectrum** finds every peak above 6 sigma and highlights the
elements whose lines would explain them. That is a stronger question than
identifying one energy, because it uses the whole pattern.

The scoring exists to reject coincidences, which at 70 eV tolerance are
everywhere. Three rules:

1. an element's **strongest visible line must be present** - a stray match on
   tin's Lb4 with no La is not tin
2. lines that **should** be visible and are not are **penalised**, which is
   what stops a dense heavy-element L series matching anything
3. matches are **weighted by peak size**, so explaining the dominant peak
   counts for more than clipping a small one

Without rule 3 a quartz spectrum ranked indium above silicon. With all three,
the top hits come out right: quartz gives O and Si, the perovskite gives Si
then Pb, I and In, salt rock gives Na and Cl, and the gold-on-carbon target
gives C and Au M.

**Treat the list as a shortlist.** Real coincidences still appear further
down - Ta M and Rb L show up in the perovskite and are not there. The fit is
what settles it, and the screening tests in section 5 settle it after that.

**Accept suggested** selects the highlighted entries. Anything not already in
the default element lists is reported in the log rather than added silently.

### 3.3 Choose elements

Three tabs — **K**, **L**, **M** — because an element can contribute lines
from more than one shell, and they must be selected separately.

Two presets:

- **Light preset** — C through Ca, K lines only
- **Silicate preset** — adds Ti–Fe and the Fe/Ni L lines

Select only what you have reason to expect. Every extra component is another
way for the fit to explain something with the wrong element.

**The live estimate.** As in GeoPIXE's fit setup, the plot redraws an
initial estimate every time the elements, the calibration or the fit range
change, with no fitting. With "Hold line shape" on it draws the held shape and
its calibration, as the held fit uses, and ignores the calibration boxes. Each element's area comes from peak heights
(`pixe_initial.pro`): for each line above 1 % of the element's major line,
the mean height above the SNIP background over +-FWHM/2, times 1.0645 x FWHM,
divided by the line's relative intensity. The smallest of these wins, so an
overlapped minor line cannot inflate it. The model is then evaluated once,
without pile-up or sum peaks. Expect it to read low: averaging over +-FWHM/2
gives about 0.8 of a pure Gaussian's height, and GeoPIXE's estimate does the
same. The plot title says **initial estimate - not fitted** while it is
showing, and FIT replaces it. It is there to show which lines
you have asked for and roughly where they land, before you spend a fit on
them. Untick **live estimate** above the plot to turn it off.

E low and E high take effect when you leave the box. They used to wait for
FIT, and that mattered. A `.detector` file loaded after typing a new range
had its resolution converted at the old E low (the width is referenced to E
low), and the fit then ran at the new one.
Since the fix for that (BUGS.md, 2026-10-19) the order no longer matters:
changing E low re-references the width so its FWHM at every energy stays
put, as GeoPIXE does when it rebuilds the width from the detector's w0, w1.

### 3.4 Fit

**Ctrl+F**, or the FIT button.

The spectrum panel shows data, background, model, and each component. **The
panel underneath is the residual in units of sigma, and it is where you should
look first.** A good fit scatters within ±2σ. A localised excursion of 5σ or
more means something in the model is wrong — and if you add an element to
absorb it, you have hidden the problem rather than solved it.

Tick **escape + pile-up curves** to draw the fit's escape lines and its pile-up
terms as two summed curves over the background, as GeoPIXE's Display > Escapes
and Pileup overlays, with or without the element components. They are the
fit's own (escape lines tied to each parent's fitted area), where GeoPIXE
builds its overlays from the measured data. The legend gives their counts
inside the fit range. The escape curve sums every element's escapes, including
those of groups hidden with **Lines...**. **Show escape + sum peaks** still
marks where each escape and sum peak falls, including the Ge K-beta escapes.

Reduced χ² appears in the legend and status bar. Typical values on real
spectra here are 1.8–2.5; below about 1.5 with high counts usually means you
have too many free components.

**How long it takes, and how to make it short.** Each element you tick is one
component, except an M shell, which is five - M1 to M5 are resolved lines at
these energies, not one peak. The perovskite cell in section 4b is 22 element
groups, which with pile-up is 50 components and about **40 s** of free
refinement on an ordinary laptop. The fit runs in the background. The status
bar shows a progress bar and a **Cancel** button, and the plot can still be
zoomed and read. Everything that would change the data or the options under
the running fit is greyed out until it finishes: the left panel, the element
table and its presets, the map tools, and the File and Analysis menus apart
from Save spectrum image. The fit uses the calibration and options as they
stood when you pressed FIT. A value typed into a box but not yet entered is
not picked up. Cancel stops the fit at its next step and keeps no result. Batches run the same way. A cancelled
batch keeps the runs it finished, and its summary lists only those.

**Hold line shape from last fit** (Fit options) refines nothing non-linear -
no calibration, width, tail, shelf or sum deficit - and solves only the areas,
starting from the shape the last fit converged to. On that same spectrum:
40.8 s free, **4.5 s held**, with identical chi2 (2.958) and areas identical
to the digit (GUI, before the L-line fix and the 2026-10-20 speedups; the free fit of the
287424 golden recipe now takes about 18 s, `tools/golden_dump.py`, 2026-10-20).

Speed is the smaller reason to use it. The real one is that a region fit must
not be allowed to re-refine the shape. Two masks over the same pixels of
run 287424 - raw counts agreeing to 0.003% - gave silicon 4.5% apart and
Pb M5 14% apart with free refinement, because the optimiser took a different
path to a shape it was never entitled to change. **Fit the whole field free,
then tick this and fit the regions.** Then a difference between two regions is
a difference in the data.

Unticked, **every fit starts from the detector width and the calibration
boxes**, never from the last fit, so a fit with the wrong elements cannot
spoil the next one. Before 2026-10-21 each fit started from the last one's
shape: on 287424 the recipe fitted to chi2r 2.382, dropping Al, Si and C gave
359.4, and putting them back gave 8.411 instead of 2.382 - started from the
misfit's shape (noise 20.05 channels) it converged to noise 11.71, and Ag L
read 7051 counts instead of 26946. Now the same clicks give 2.382, 48.15 and
2.382 again, with the first fit's shape (headless MainWindow driving those
clicks, 2026-10-21). GeoPIXE also starts
each fit from its detector file (`pixe_setup.pro:106-107`), though it writes a
free fit's width back into the detector for the next fit
(`pixe_cleanup.pro:75-80`); scrappyFIT does not. **Reset line shape** forgets
the last fit's shape, so a held fit refines once from the detector again;
opening another file does the same. The live estimate draws the shape the
next fit starts from: the held one while the hold is on, else the detector's.

### 3.5 Map an element, then mask a region

This is the part that matters most and the part GeoPIXE makes hardest.

Go to the **Maps** tab, choose a line, **Show map**. The list holds the lines
inside the fit range: `Si` is the K line, `Ag L` and `Pb M` the L-alpha and
M-alpha lines; typing `AgL`, `Ag:L`, `ag l` or an energy in keV works too. A line whose
peak window is off the spectrum is refused (the flanking background windows
are not checked yet, BUGS.md), and so is **Mask above percentile** on a
flat map (every pixel would be "above" it). Map browsing never changes the
spectrum or the fit; only a mask does, and the title then carries its name.

Then either:

- **drag a rectangle** on the map, or
- choose **polygon** as the mask tool and click its corners (click the first
  corner again to close it), or
- set a percentile and press **Mask above percentile**

Either way the spectrum becomes the sum over just those pixels. Refit, and you
are analysing that region rather than the whole field.

The active mask is drawn on the map (light tint, cyan outline). **grow** and
**shrink** move its edge by one displayed pixel (the binning: 4 raster pixels
at the default) whichever tool made it, so a rectangle stays a rectangle.
Set the mask op to **add** or **subtract** to build a region from several
shapes (a grain plus its rim, or a grain minus a crack). The op applies to every
mask tool, percentile and flood fill included, but not to grow / shrink.
Subtract with no mask set cuts the shape out of the whole field.
Zoom and pan with the toolbar above the map; **Reset view** returns to the whole
field and the default contrast and keeps the mask.
The row below sets the colour table, a linear / log / sqrt scale, and the
colour limits as percentiles of the map (2 and 99 by default); they carry over
when you show the next element. The arrows beside the element list step to the
previous / next element and map it. **Export map...** writes the map as CSV, and
the mask beside it if one is set.

Back on the **Spectrum** tab the legend names the region: its mask name, its
size in raster pixels (a binning-4 map pixel is 16 of them) and its total
counts. Tick **whole field** to draw the whole-field spectrum behind it, scaled
by the region's share of the scanned pixels, so equal heights mean equal counts
per pixel (not equal charge: collection varies across a field). A mask cut by
thresholding an element's map is biased for that element's own line; compare the
other lines, or cut on half the events and measure on the other half. The
**legend** box picks what the legend lists: **compact** (data, fit, background,
pile-up and escape; the default), **all** (every drawn element component too)
or **off**. The legend sits to the right of the plot, beside both panels, and
never over the data: it takes as many columns as its length needs, and the plot
narrows by its width. In a small window, once it would take more than 40% of the
width, the last element entries fold into one "+ n more" line. **Lines...**
picks which element groups (Fe K, Ag L, Ag M, ...) are drawn and listed, with
All and None buttons; the choice is kept across fits and changes only the
display, never the fit.

**Why this matters.** The bulk spectrum of a heterogeneous sample describes
nowhere in it. A real case from the development of this program: a salt-rock
scan gave 5.5 µg/cm² of surface carbon. Masking on the carbon map showed the
carbon was confined to about 22% of the field, at 16 µg/cm², with essentially
zero over the remaining bare salt. The bulk number described neither region.
Splitting the field also took reduced χ² from 1.99 to 0.75 and 1.34, because
one model genuinely cannot describe two different materials at once.

**A warning about window maps.** A net map subtracts flanking background, and
that fails where a weak line sits between two strong ones. Aluminium between
magnesium and silicon is the standard case: the map goes negative and is
meaningless. scrappyFIT warns you when a net map comes out mostly negative.
Untick *subtract background* and read raw counts, or use a DA matrix map,
which resolves overlaps properly.

### 3.6 Export

**Ctrl+E**, choose a folder. You get:

| file | contents |
|---|---|
| `*_spectrum.txt` | channel, energy, counts |
| `*_background.txt` | the SNIP background |
| `*_fit_model.txt` | the fitted model |
| `*_components.txt` | every component separately, one column each |
| `*_residual.txt` | residual in sigma |
| `*_areas.txt` | areas with uncertainties |
| `*_mask.npy` | the mask, if one was applied |
| `*_analysis.json` | **every option used**, the calibration, the charge, the database path, and the scrappyFIT version |

That last file is the important one. A number without the settings that
produced it is not reproducible, and the JSON means a folder opened in a year
still says what was done.

### 3.7 The sample's yield files

GeoPIXE quantifies every element from one column of the yield table, the
layer marked UNKNOWN (`pixe_fit.pro:414`, `conc = area / yield` at `:767`).
It has no per-element layer choice, so a layered sample needs one `.yield`
per UNKNOWN layer, and one set per region when the stack changes across the
map. Run 287424's are named for that,
`pvsk287424_<region>_UNKNOWN-<layer>.yield` (regions `bare`, `plateau`,
`void`; layers `Ag`, `perovskite`, `ITO`, `SnO2`, `glass`), and scrappyFIT
reads region and layer from names of that form. Any other `.yield` is a set
of one, and its layer is read from the file.

**Where the files come from.** When a file is opened, scrappyFIT looks in
this order and stops at the first hit:

1. the files remembered for this data file (every choice made in Sample
   model is remembered);
2. files beside the data whose name contains the data file's name
   (`287424.lmf` → `pvsk287424_bare_UNKNOWN-ITO.yield`), or the one
   `.yield` that the old rule accepts (same stem, or a `.provenance.json`
   naming this data file);
3. the same name rule in each folder of the search path (**Sample model >
   Search folders...**).

The remembered choices and the search path are kept in
`~/.scrappyfit/yields.json`, or in the file `SCRAPPYFIT_YIELD_STORE` names.
A file whose `.provenance.json` says its yields are in RELATIVE units is never
picked up: it cannot give a concentration. The load message says what was
found, or why nothing was.

**Sample model** shows the set at the top. It lists the folder, how the set
was found, and the file for each layer of the chosen **region**. **Folder...**
and **Files...** choose a set, and **Search folders...** edits the path.
**Quantify with these .yield files** is selected whenever a set is found or
chosen. A file just written by **File > Export yield** is the exception: it
is the model's own yield, so the one-layer QUANTIFY stays the default for it.
**Analysis > Layer stack and escape** opens on a file of the region in use,
with its UNKNOWN layer selected. A list at the top draws any other file of
the set; this only draws it and changes nothing QUANTIFY reads. Right after
**File > Export yield** it shows the exported file, which leaves the set
QUANTIFY reads untouched. **Copy its
composition** still fills the one-layer model from the file shown.

**QUANTIFY with yield files** takes GeoPIXE's column and normalisation and
scrappyFIT's total-shell photons. From each file it reports only the elements
that live in that file's UNKNOWN layer. An element that lives in two layers
(Sn in SnO2 and in ITO; O in glass, ITO and SnO2) gets a row from each, and
each row gives the whole signal to that one layer. Those rows are alternatives
and must not be added: quote one layer, the one the element really lives in
(nearly all O K comes from the glass). A fitted element that lives in no
UNKNOWN layer is listed as not quantified. The formula is

    ppm (by weight, in that layer) = (area / eff(major line))
                                     / (Q x Omega x Y_major x sum of the file's line intensities)

Q is the entered charge x live fraction. With a region mask applied it is
also multiplied by the masked fraction of the pixels, so the region gets its
share of the field's charge. Without charge and solid angle the values are
ppm x uC x msr, and ratios within one layer still hold. The pixel fraction
assumes uniform dose over a fully scanned raster (true of 287424, where the
RBS dose is flat to 1 %). When the M subshells are fitted apart, all of them
are counted, unless the file's `.provenance.json` says its M yield is the
M5 + M4 share only (key `pb_m`, as in the lab's 287424 files). Such a file
cannot be used with a fit that lumped the M shell into one component: the row
says so and gives no number.

This agrees with GeoPIXE's own `area_major / (Y x eff)` when scrappyFIT's
major-line fraction equals 1 / (the sum of the file's absorption-weighted
intensities). The two differ most for L lines under an overlayer and for M
shells. The output names every file used. With the bootstrapped or typed
one-layer model, the output says so and why ("bootstrapped ... because no
.yield named for ... on the search path").

---

## 4. The physics

### 4.1 Mass attenuation coefficients

Selectable at run time. Five choices:

| dataset | range | notes |
|---|---|---|
| `henke1993` | 10 eV – 30 keV | CXRO f2. Full light-element coverage |
| `ffast` | 10 eV – 30 keV | Chantler, NIST. Full coverage |
| `xcom` | 1 keV up | Berger & Hubbell. **Nothing below 1 keV** |
| `sabbatuccisalvat2016` | 1 keV up | **Nothing below 1 keV.** Not shipped (EGSnrc data, AGPL-3.0): fetch it first, see below |
| `mixed` | full | **the default** |

`mixed` implements the rule measured by Heirwegh (2014): **FFAST below each
absorber's own K edge and everywhere below 1 keV; XCOM above the edge.** On
silicate standards that selection took the pure-element to silicate-oxide
discrepancy from 6–8% down to under 2%, and under 1% for Mg and Si.

Benchmarked here against a quartz sample of exactly known composition
(SiO₂, O 53.26 / Si 46.74 wt%), scored as mean |ln(measured/true)|:

| dataset | O wt% | Si wt% | score |
|---|---|---|---|
| Henke 1993 | 52.36 | 43.17 | 0.0482 |
| FFAST | 50.14 | 45.39 | 0.0448 |
| **mixed** | 49.55 | 46.02 | **0.0439** |
| XCOM alone | — | — | cannot return oxygen |

**Do not select Sabbatucci or XCOM for light-element work.** They cannot
return a value for C, N, O or F at all.

The Sabbatucci-Salvat table is converted from EGSnrc's `photo_shellwise.data`,
which is AGPL-3.0, so scrappyFIT does not ship it (v0.25.0 was the last
release that did). Selecting it without the table is an error that says how to
get it. To install it once, run

```
python -m scrappyfit.io.sabbatucci
```

which downloads EGSnrc's file (pinned commit, SHA-256 checked), converts it and
writes `MAC_SabbatucciSalvat2016.txt` to `~/.scrappyfit/data`
(`SCRAPPYFIT_DATA_DIR` overrides the folder; `--source FILE` converts a copy you
already have). Choosing it in the GUI's MAC box offers the same download.

### 4.2 Line data

Two options: the shipped `xray_lines.txt`, or `xray_lines_rebuilt.txt`, whose
L and M sections come from xraylib (Campbell & Wang Dirac-Fock rates). **K
lines are bit-identical between the two.**

What the rebuild fixes, as a fraction La : Lb : Lℓ : Lη:

| element | original | rebuilt |
|---|---|---|
| Ca | 0.893 : 0.089 : 0.009 : 0.009 | **0.000** : 0.193 : **0.537** : 0.270 |
| Sc | 0.893 : 0.089 : 0.009 : 0.009 | **0.000** : 0.106 : **0.594** : 0.299 |
| Ti | 0.893 : 0.089 : 0.009 : 0.009 | 0.627 : 0.285 : 0.069 : 0.019 |
| Fe | 0.766 : 0.157 : 0.061 : 0.015 | 0.669 : 0.240 : 0.071 : 0.019 |

Lℓ now falls smoothly with Z as the 3d shell fills, which is the physics. Also
fixed: At/Rn/Ac had Lℓ and Lη set to exactly zero, and Mα1:Mα2 was an exact
50:50 split for every element (the real ratio is about 0.94:0.06).

Coster–Kronig factors come from xraylib rather than Elam. Elam reports f23 of
0.25 (Ti), 0.42 (Fe), 0.45 (Ni); that transition is energetically forbidden
below about Z = 30 and xraylib correctly gives zero.

### 4.3 Detector response

GeoPIXE's model is a Gaussian plus one low-side exponential tail. scrappyFIT
adds two terms, both following Heirwegh (2014):

**Si LVV Auger escape step.** A silicon Auger electron escapes the crystal
carrying about 90 eV. Extends below each peak by `max(90 eV, 8% of E)` — the
fixed term dominates below 1.1 keV, which is the whole light-element range.

**Contact injection step.** A photon absorbed in the window metallisation
ejects an electron into the crystal. Edge at the Al L3 binding energy,
72.7 eV. On the Amptek C1 window (90 nm Si₃N₄ over **250 nm Al**) this is
substantial; the C2 window carries only 30 nm of Al and should show far less.

Both use a **sloped** difference-of-exponentials profile, not a flat step,
because measurement shows these shelves slope.

A Si₃N₄ window term (N K edge, 409.9 eV) is implemented but fits to zero on
this detector. It is off by default.

What they buy, on a chondrite:

| model | χ²ᵣ |
|---|---|
| Gaussian + tail only | 2.468 |
| + Si LVV escape | 1.959 |
| + Al contact injection | 1.890 |
| + sloped shelves | **1.814** |

More importantly, the cost of *removing* the spurious titanium fell from 0.415
to 0.076 in χ². The artefact is now largely explained by detector physics
rather than by inventing an element.

**The shelf slopes are fixed, deliberately.** Fitting them per spectrum
improves χ² by about 0.02 and returns values that do not transfer between
samples — 3.0/0.2 from a chondrite, 12.0/3.4 from a quartz, each worse than
the default on the other. A genuine detector property must transfer. Heirwegh
could fit them because his spectra were monochromatic with 10⁷–10⁸ counts in a
single peak.

### 4.4 Background

SNIP (Ryan et al. 1988), ported line for line from `strip_clip.pro`, including
the statistics-sensitive `low_stats_filter` that runs before it. Omitting that
filter made the background 20–45% low on a real spectrum, an error that lands
almost entirely on the weak peaks — which is where trace analysis lives.

### 4.5 Yields

Thick-target integration following Reuter et al. (1975): Andersen–Ziegler
stopping power, Cohen & Harrigan K-shell ionisation cross sections, Krause
fluorescence yields with optional Elam substitution below Z = 10, secondary
fluorescence, and layered samples with per-layer attenuation.

**On fluorescence yields for light elements:** three independent lines of
evidence in this work favour Krause over Elam — a NaCl stoichiometry test,
differential absorption on a carbon patch, and the quartz benchmark. Elam is
available but is not the default below Z = 10.

---

## 4b. Charge, solid angle, and why the total matters

### The one number that can fail

Normalising to 100 wt% is the default because it cancels three things that are
awkward to know: the solid angle, the charge, and the constant buried in
atoms per gram. It is also a way to be confidently wrong. A fit that has lost
a third of its intensity, or that is missing an element outright, still sums
to exactly 100% afterwards and looks healthy.

Measured on run 287427, a quartz, with everything else held fixed:

| case | chi2 | normalised sum | absolute sum |
|---|---|---|---|
| everything modelled | 1.92 | 100.0% | 100.0% |
| oxygen removed | 1385 | 100.0% | 32.4% |
| aluminium removed | 2.71 | 100.0% | 97.2% |
| shelf, tail, escape, pile-up off | 2.55 | 100.0% | 97.7% |

The normalised column is 100.0 in every row, including the row where a major
element is simply not in the model. Note also what normalising does to the
survivors when oxygen is dropped: silicon goes from 41.7 to 62.4 wt% and
fluorine from 0.40 to 36.8 wt%, both entirely spurious, and both reported
with small fitting errors.

In fairness, chi2 catches the oxygen case on its own. What chi2 cannot catch
is a **scale** error, because the shape of the model is untouched:

| case | chi2 | normalised sum | absolute sum |
|---|---|---|---|
| 20% dead time, not corrected | 1.922 | 100.0% | 125.0% |
| charge digitiser reading 15% high | 1.922 | 100.0% | 87.0% |
| detector really at 33 mm, assumed 30 | 1.922 | 100.0% | 121.0% |

Identical chi2 to three decimal places, identical normalised result, and
three different wrong answers. Only the absolute total moves. That is the
argument for filling in the geometry panel.

### Filling it in

`counts = concentration x yield x Q x Omega x live`, so

  - **charge, uC** - integrated beam dose. The weakest link. Current read on
    the sample rather than in a suppressed Faraday cup runs high, sometimes
    by tens of percent, because secondary electrons leaving the target count
    as beam current arriving. Get this from a Faraday cup if at all possible.
  - **distance, mm** - sample to detector face. Enters as 1/r^2, so at a
    30 mm working distance 3 mm of error is 21%. Measure it, do not take it
    from the drawing.
  - **active area, mm2** - as the datasheet quotes it (Amptek: 25, 70). If
    there is a collimator, the collimator's area is the one that counts.
  - **dead time, %** - counts never recorded. This is not in the LMF (see
    below), so it has to come from the OMDAQ run log. Ignoring it makes every
    concentration low by the same factor: invisible normalised, obvious
    absolutely.

The panel shows the solid angle as soon as distance and area are both filled
in. Seeing a plausible number appear - a few tens of msr for a close-coupled
detector - is the cheapest check that neither was mistyped.

### Let the detector file fill it in

Two of those four numbers are already written down. A GeoPIXE `.detector`
file carries the crystal diameter and the source distance, so
**Load .detector file...** (in the Detector efficiency panel, or `--detector`
on the command line) now fills the distance and active area boxes and the
solid angle appears without anything being typed. On the C1 deflector
detector: diameter 4.652 mm is 17.00 mm2, at 77.525 mm, so 2.829 msr.

This matters beyond saving two keystrokes. The same file supplies the
efficiency absorbers, the crystal, and the energy-dependent tail, so a
hand-entered solid angle that disagrees with it makes the absolute total
wrong in a way the normalised total hides - exactly the failure the table
above is about. Note that 17 mm2 is what the geometry of *this* detector
gives; a nominal "25 mm2" from the datasheet would overstate the solid angle
by 47%.

Values already in the boxes are never overwritten, so a measured distance or
a collimated area survives loading the file. Clear a box to let the file
supply it again.

### What the LMF does and does not give you

`lmf.clocks()` reads the per-block header. Field 0 is a dose counter, and it
is **digitiser pulses, not microcoulomb**; the conversion depends on the
current-digitiser range in use, which the file does not record. So it gives
relative charge exactly - useful for comparing two runs from one session -
and absolute charge only once someone supplies uC per count.

Fields 1 to 4 are four free-running clocks in microseconds. They track each
other to within a few hundred microseconds across a 40-minute run, which
rules out reading them as elapsed versus live: a live clock would fall behind
by percent, not by microseconds. **Dead time is therefore not recoverable
from the LMF.** `clocks()` returns `live_fraction = None` rather than
inventing 1.0, because a fabricated 1.0 biases every absolute concentration
low without saying so.

### The honest route when charge is not trustworthy

If the charge cannot be trusted - and often it cannot - do not abandon
absolute work; move the calibration to something you can measure. Run a
standard of known composition, use `Session.instrument_constant()` to get
counts per unit yield from one element whose concentration you know, and
apply that constant to the unknowns from the same session. That is what a
standard is for, and it is why a measurement against one is worth more than
an absolute calculation from first principles.

### Reading the total

| total | reading |
|---|---|
| 95-105% | consistent; the chain holds together |
| below 90% | intensity unaccounted for: a missing element, uncorrected dead time, or charge reading high |
| above 110% | something counted twice, charge reading low, or solid angle overstated |

A total near 100% is not proof the analysis is right - two errors can cancel.
It is a test the analysis can fail, which is more than a normalised total
will ever offer.

## 5. Reading a result critically

### The screening tests

Three physical checks are available for deciding whether a fitted L or M
assignment is real:

**Coster–Kronig floor.** Vacancies created in L1 and L2 are forced into L3 at
a known rate. A fit that fills L1/L2 while leaving L3 empty describes
something that cannot happen.

**M/L corroboration.** When both are in range, M intensity is bounded by L
intensity through the fluorescence yields. A silver M line at 215 times its
physical bound is not silver.

**K-vs-L agreement.** For Ti–Ni both shells are inside a 6.6 keV window, so
they must give the same answer. Predicted L/K rises smoothly with Z — 0.43 for
Ti, 3.31 Mn, 6.35 Fe. Measured on a chondrite: Fe 3.41 (agrees within 2×), Mn
20.7 (6× out), Ti 12.0 (**28× out**).

### The rule of thumb

**Any element whose only visible line sits 40–90 eV below a strong
light-element peak should be treated as suspect until another shell confirms
it.** In practice that means Ti L, Cr L, Mn L, and the sub-keV M lines of Ag,
Sn and In.

### How to tell an artefact from an element

An artefact **follows its parent peak**. Measured across five samples, the
excess sits about 60 eV below whatever peak produced it:

| parent | excess at | offset |
|---|---|---|
| C 0.277 | 0.242 | −35 eV |
| O 0.525 | 0.457 | −68 eV |
| Si 1.740 | 1.703 | −37 eV |
| S 2.308 | 2.208 | −100 eV |
| Fe 6.404 | 6.323 | −81 eV |

A real element sits at one fixed energy regardless of what else is in the
sample.

---

## 6. Scripting

The GUI is a thin layer over `Session`, which is equally usable directly:

```python
from scrappyfit.session import Session

s = Session()
s.load('run.lmf')
s.set_calibration(0.0017019, -0.3784)

s.run_fit(['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'K', 'Ca', 'Fe'])
print(s.fit.reduced_chi2, s.areas())

carbon = s.element_map('C', binning=8)
s.set_mask(carbon >= np.percentile(carbon, 78), 'C-rich')
s.run_fit(['C', 'N', 'O', 'Na', 'Cl'])

s.export('out/')
```

Options are set on `s.options` before fitting. Anything that changes a result
lives there and is written into the exported JSON.

---

## 7. Limits and known gaps

- The **binary** GeoPIXE `.spec` format is not read; only the text export
- No OMDAQ live connection yet
- Shelf slopes are fixed (see 4.3)
- Concentration conversion needs a detector efficiency curve, which must be
  supplied; there is no built-in model of your detector's window
- The response terms were fitted on an Amptek X-123 with a C1 window and a
  500 µm crystal. **Refit them for a different detector** — the contact term
  in particular scales with the window's aluminium thickness
- Elemental maps assume a 256×256 raster
- **`LayeredYieldModel.yields()` is K-shell only.** There are no L or M
  proton-ionisation cross sections in the vendored database, so an element
  fitted on its L or M lines returns a yield of zero rather than a wrong one.
  It is deliberately absent, but it means QUANTIFY cannot report silver,
  indium, tin, iodine, caesium or lead from this stack - only their areas.

---

## 8. References

- Ryan, Clayton, Griffin, Sie, Cousens (1988), *SNIP*, Nucl. Instr. Meth. B34, 396
- Reuter, Lurio, Cardone, Ziegler (1975), J. Appl. Phys. 46, 3194
- Elam, Ravel, Sieber (2002), Radiat. Phys. Chem. 63, 121
- Chantler (1995, 2000), NIST FFAST
- Berger & Hubbell, XCOM
- Henke, Gullikson, Davis (1993), At. Data Nucl. Data Tables 54, 181
- Scholze & Procop (2009), X-Ray Spectrom. 38, 342
- **Heirwegh (2014)**, *Studies of Light Element X-Ray Fundamental Parameters
  Used in PIXE Analysis*, PhD thesis, University of Guelph — the source for
  the detector response decomposition and the mixed MAC rule
- Ryan et al., GeoPIXE, CSIRO
