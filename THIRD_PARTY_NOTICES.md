# Third-Party And Data Notices

The Apache-2.0 license in `LICENSE` applies to Acoustic Agent source code and
project-authored documentation. Bundled datasets retain their own terms. The
metadata embedded in each source file is authoritative.

## CIPIC HRTF Database

File: `acoustic_agent/resources/hrtf/cipic_124.sofa`

- Database: CIPIC HRTF Database, subject 124.
- Owner/organization: The Regents of the University of California.
- Reference: V. R. Algazi, R. O. Duda, D. M. Thompson, and C. Avendano,
  "The CIPIC HRTF Database," WASPAA 2001, pp. 99-102.
- Terms: the SOFA `License` attribute grants reproduction and use for
  educational, research, and commercial purposes, requires preservation of
  any present copyright notice, and requests acknowledgment in publications
  and notification for commercial-product use.
- Conversion: converted to SOFA by Piotr Majdak, Acoustics Research Institute,
  Austrian Academy of Sciences.

## SADIE II HRTF Database

File: `acoustic_agent/resources/hrtf/sadie_h12.sofa`

- Database: SADIE II, human subject H12.
- Copyright: 2018 University of York.
- License: Apache License 2.0, as recorded in the SOFA `License` attribute.
- Reference: https://doi.org/10.3390/app8112029

## Acoustic Materials V3

File: `acoustic_agent/resources/acoustic_materials/acoustic_materials_v3.sqlite3`

This project-compiled runtime database contains 3,741 six-band records, 16
material families, 20 semantic categories, and 64 semantic-material mappings.
The open-source runtime repository distributes the compact SQLite database, not
the offline authoring workspace or duplicate intermediate exports.

The material database has mixed upstream terms and uses the SPDX value
`NOASSERTION`. Apache-2.0 covers the project-authored schema, semantic taxonomy,
VLM-assisted mappings, quality metadata, build scripts, and sampler, subject to
rights in the underlying records. It does not relicense all coefficient data.
See `acoustic_agent/resources/acoustic_materials/DATA_LICENSE.md` and
`sources.json` for the row counts, URLs, and redistribution status summarized
below.

### PTB Absorption Coefficient Database

- Records: 2,573.
- Publisher: Physikalisch-Technische Bundesanstalt (PTB).
- Source: https://www.ptb.de/cms/en/ptb/fachabteilungen/abt1/fb-16/ag-163/absorption-coefficient-database.html
- Terms: the official page states that the Excel database can be downloaded
  and used as desired. No standard SPDX license is identified. Preserve PTB
  attribution and verify the current source terms when redistributing.

### ODEON And Manufacturer Material Sheets

- Records: 981.
- Material portal: https://odeon.dk/downloads/materials/
- Supplier format guide:
  https://odeon.dk/download/materials/Material%20sheet%20guide%20for%20suppliers.pdf
- Website terms: https://odeon.dk/disclaimer/
- Terms: no blanket open-data redistribution grant was identified. The source
  spreadsheets include manufacturer-specific data, and rights may belong to
  ODEON and/or the named manufacturers. Written permission is required before
  publicly redistributing this subset.

### Pyroomacoustics Materials Database

- Records: 90.
- Documentation:
  https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.materials.database.html
- Repository and MIT license: https://github.com/LCAV/pyroomacoustics
- Citation: R. Scheibler, E. Bezzam, and I. Dokmanic, "Pyroomacoustics: A
  Python package for audio room simulations and array processing algorithms,"
  Proc. IEEE ICASSP, 2018.
- Terms: MIT. Preserve the upstream copyright and permission notice. Individual
  coefficient entries may also carry literature references in Pyroomacoustics.

Upstream MIT notice:

> Copyright (c) 2014-2017 EPFL-LCAV
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

### Acoustic Supplies Coefficient Chart

- Records: 67.
- Publisher: JCW Acoustic Supplies.
- Source: https://www.acoustic-supplies.com/absorption-coefficient-chart/
- Terms: the chart is publicly readable, but no open-data redistribution
  license was identified and the site displays an all-rights-reserved notice.
  Written permission is required before publicly redistributing this subset.

### SoundSpaces Acoustic Material Presets

- Records: 30.
- Project: https://github.com/facebookresearch/sound-spaces
- Source file:
  https://github.com/facebookresearch/sound-spaces/blob/main/scripts/mp3d_acoustic_properties.py
- License: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://github.com/facebookresearch/sound-spaces/blob/main/LICENSE
- Citations: C. Chen et al., "SoundSpaces: Audio-Visual Navigation in 3D
  Environments," ECCV 2020; and C. Chen et al., "SoundSpaces 2.0: A Simulation
  Platform for Visual-Acoustic Learning," NeurIPS Datasets and Benchmarks 2022.
- Change notice: Acoustic Agent normalized the presets, mapped them into a new
  semantic taxonomy, derived sampling classes, and added quality metadata.
  These changes are not endorsed by the SoundSpaces authors.

### Acoustic Agent Transformations

The project used VLM-assisted Semantic-to-Material Mapping to classify material
names and descriptions. It then applied deterministic rules and QA checks to
organize 20 simulator-facing semantics, 16 material families, four absorption
classes, and six octave bands. The VLM did not generate absorption coefficients.
Original source-group identifiers are retained on runtime material rows.

Do not call the complete database "open source" or "Apache-2.0 data." A public
release must obtain permission for the ODEON/manufacturer and Acoustic Supplies
subsets or exclude them through a reviewed redistributable build profile.

## ResPlan V1

File: `acoustic_agent/resources/resplan/resplan_v1.sqlite3`

This project-compiled database contains 15,376 audited, single-floor scenes
derived from the local `ResPlan.pkl` source. Records are stored as zlib-compressed
JSON while preserving the source scene index used by the API.

The source dataset currently has no standalone license file in this repository.
The repository owner must confirm its provenance and redistribution rights
before public release. Until that review is complete, do not assume Apache-2.0
applies to the database contents.
