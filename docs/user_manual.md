# scrappyFIT user manual

Version 0.1.0

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

### From a checkout

```bash
pip install -e .
```

### Dependencies

`numpy`, `scipy`, `matplotlib`, `PyQt5`, `xraylib`, `xraydb`. All available
from PyPI.

### The atomic database

scrappyFIT needs GeoPIXE's data directory — the one containing `dat/` with
`ElamDB12.txt`, `xray_lines*.txt`, `MAC_*.txt`, `xsect_K/L/M.txt` and
`hubbell.dat`. It is found in this order:

1. a path passed to `config.set_database_path()`
2. the `SCRAPPYFIT_DB` environment variable
3. a copy vendored at `scrappyfit/resources/database`
4. the path saved in `~/.scrappyfit/config.json`
5. a GeoPIXE tree sitting beside the checkout

In the GUI: **File → Set database folder**. It is remembered.

For a flash-drive or offline install, copy the database into
`scrappyfit/resources/database` and it travels with the code.

### Running

```bash
python -m scrappyfit
```

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

### 3.3 Choose elements

Three tabs — **K**, **L**, **M** — because an element can contribute lines
from more than one shell, and they must be selected separately.

Two presets:

- **Light preset** — C through Ca, K lines only
- **Silicate preset** — adds Ti–Fe and the Fe/Ni L lines

Select only what you have reason to expect. Every extra component is another
way for the fit to explain something with the wrong element.

### 3.4 Fit

**Ctrl+F**, or the FIT button.

The spectrum panel shows data, background, model, and each component. **The
panel underneath is the residual in units of sigma, and it is where you should
look first.** A good fit scatters within ±2σ. A localised excursion of 5σ or
more means something in the model is wrong — and if you add an element to
absorb it, you have hidden the problem rather than solved it.

Reduced χ² appears in the legend and status bar. Typical values on real
spectra here are 1.8–2.5; below about 1.5 with high counts usually means you
have too many free components.

### 3.5 Map an element, then mask a region

This is the part that matters most and the part GeoPIXE makes hardest.

Go to the **Maps** tab, choose an element, **Show map**.

Then either:

- **drag a rectangle** on the map, or
- set a percentile and press **Mask above percentile**

Either way the spectrum becomes the sum over just those pixels. Refit, and you
are analysing that region rather than the whole field.

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

---

## 4. The physics

### 4.1 Mass attenuation coefficients

Selectable at run time. Five choices:

| dataset | range | notes |
|---|---|---|
| `henke1993` | 10 eV – 30 keV | CXRO f2. Full light-element coverage |
| `ffast` | 10 eV – 30 keV | Chantler, NIST. Full coverage |
| `xcom` | 1 keV up | Berger & Hubbell. **Nothing below 1 keV** |
| `sabbatuccisalvat2016` | 1 keV up | **Nothing below 1 keV** |
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
