# Third-party notices

scrappyFIT's own code is MIT-licensed, Copyright (c) 2026 University of North Texas
(see `LICENSE`). The atomic data it bundles in `scrappyfit/resources/` comes from the
sources below, and each file keeps the terms of its source.

## CSIRO GeoPIXE database (MIT, Copyright (c) 2023 CSIRO)

These files are byte-identical copies (git blob hashes compared on 2026-10-20) of the files of
the same name in CSIRO's open-source GeoPIXE, <https://github.com/CSIRO-GeoscienceAnalytics/GeoPIXE>, directory
`Workspace/main/database/dat`. That repository is licensed under MIT, Copyright (c) 2023
CSIRO. The licence does not grant any right to use CSIRO's trade marks or logos. CSIRO's
licence text ships verbatim beside the files as
`scrappyfit/resources/database/dat/LICENSE-CSIRO-GeoPIXE`.

| file | content | original data source (cite these) |
|---|---|---|
| `ElamDB12.txt` | edges, fluorescence yields, Coster-Kronig factors, line branching | Elam, Ravel & Sieber, Radiat. Phys. Chem. 63 (2002) 121 |
| `xray_lines.txt` | GeoPIXE's flat X-ray line table | GeoPIXE |
| `xsect_K.txt`, `xsect_L.txt`, `xsect_M.txt` | proton ionisation cross sections | GeoPIXE |
| `proton.txt`, `alpha.txt` | proton and helium electronic stopping-power coefficients | GeoPIXE (Andersen-Ziegler form) |
| `hubbell.dat` | photon mass attenuation (XCOM class) | Berger & Hubbell |
| `edge.txt`, `mass.txt`, `element.txt` | edge energies, atomic masses, element symbols | GeoPIXE |

## Files produced for scrappyFIT

| file | source | terms |
|---|---|---|
| `MAC_FFAST.txt` | NIST FFAST photo-absorption (C. T. Chantler et al.), read through the `xraydb` package by `tools/make_ffast.py` | FFAST is a work of the US National Institute of Standards and Technology and is not subject to copyright in the US; cite Chantler, J. Phys. Chem. Ref. Data 24 (1995) 71 and 29 (2000) 597 |
| `MAC_Henke1993.txt` | derived from the CXRO/LBNL atomic scattering factors f2 (<https://henke.lbl.gov/optical_constants/>) | CXRO states no licence for the tables; cite Henke, Gullikson & Davis, At. Data Nucl. Data Tables 54 (1993) 181 |
| `xray_lines_light.txt` | Li, Be, B lines from Elam et al. 2002 (`ElamDB12.txt`), with the solid-state Li K energy from Llovet et al., Microsc. Microanal. 29 (2023) 540 | scrappyFIT, MIT (UNT) |
| `xray_lines_rebuilt.txt` | `xray_lines.txt` with L and M rebuilt from `ElamDB12.txt` and xraylib by `tools/rebuild_lines.py` | scrappyFIT, MIT (UNT); xraylib is BSD-3-Clause |
| `efficiency/*.txt`, `response/*.json` | efficiency curves and measured response of the UNT Ion Beam Lab's Amptek detector | scrappyFIT, MIT (UNT) |

## EGSnrc-derived file (AGPL-3.0-or-later)

| file | source | terms |
|---|---|---|
| `MAC_SabbatucciSalvat2016.txt` | total photoelectric cross sections of Sabbatucci & Salvat, Radiat. Phys. Chem. 121 (2016) 122, converted from EGSnrc's `HEN_HOUSE/data/photo_shellwise.data` (barns/atom to cm^2/g) | EGSnrc, Copyright National Research Council of Canada. EGSnrc's `HEN_HOUSE/data/README.md` says every file in that directory is distributed under EGSnrc's licence, the GNU AGPL-3.0-or-later (<https://github.com/nrc-cnrc/EGSnrc/blob/master/LICENCE.md>) |

**Open question (2026-10-20):** whether this file stays in the package is under review. It is
not used by default: the default `mixed` attenuation uses FFAST (Henke where FFAST has no
value) and XCOM. You only reach it by choosing `sabbatucci` as the MAC dataset. Until this is
decided, the file ships in the wheel under its own AGPL terms.

## Python dependencies

These are installed by pip and not bundled: numpy, scipy and matplotlib (BSD-style), PyQt5
(GPL-3.0 or Riverbank commercial), xraylib (BSD-3-Clause) and xraydb (MIT). PyQt5 is
GPL. If you distribute a frozen application that bundles it, GPL terms apply to that bundle.
