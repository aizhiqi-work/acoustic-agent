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

This is a project-compiled, lossless runtime representation of the fields used
from `acoustic_material_db_v3_20260717`. The runtime database contains 3,741
six-band material records, 16 material families, 20 semantic categories, and
64 semantic-material mappings. The companion CSV, JSON, and JSONL indexes are
included for inspection and reproducible rebuilding.

The source collection currently has no standalone license file in this
repository. The repository owner must confirm provenance and redistribution
rights for every source contributing to this compiled database before making a
public release. Until that review is complete, do not assume Apache-2.0 applies
to the database contents.

## ResPlan V1

File: `acoustic_agent/resources/resplan/resplan_v1.sqlite3`

This project-compiled database contains 15,376 audited, single-floor scenes
derived from the local `ResPlan.pkl` source. Records are stored as zlib-compressed
JSON while preserving the source scene index used by the API.

The source dataset currently has no standalone license file in this repository.
The repository owner must confirm its provenance and redistribution rights
before public release. Until that review is complete, do not assume Apache-2.0
applies to the database contents.
