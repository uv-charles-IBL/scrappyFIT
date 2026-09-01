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
