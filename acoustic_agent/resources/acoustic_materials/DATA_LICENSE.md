# Acoustic Materials DB: Data Terms

This file applies to the material records and their derived indexes under this
directory. It does not replace the Apache-2.0 license for Acoustic Agent source
code and project-authored documentation.

## Mixed-license status

Acoustic Materials DB is a compilation of records from several upstream
sources. The compilation therefore uses the SPDX value `NOASSERTION`: there is
no single license that safely describes every row. In particular, the bundled
material records are not relicensed as Apache-2.0 merely because they are
distributed with Apache-2.0 software.

The project-authored schema, semantic taxonomy, VLM-assisted mappings, quality
metadata, build scripts, and sampling implementation are licensed under the
repository's Apache-2.0 license, subject to any rights in the underlying data.

| Source group | Records | Upstream terms | Public redistribution |
| --- | ---: | --- | --- |
| PTB | 2,573 | Official-page permission; no standard SPDX license | Preserve attribution and recheck the source page |
| ODEON / manufacturers | 981 | No blanket open-data license identified | Written permission required |
| Pyroomacoustics | 90 | MIT | Allowed with the upstream notice and citations |
| Acoustic Supplies | 67 | No open-data license identified; site states all rights reserved | Written permission required |
| SoundSpaces | 30 | CC BY 4.0 | Allowed with attribution, license link, and change notice |

Machine-readable source details and URLs are in `sources.json`. Full attribution
and citations are in the repository root `THIRD_PARTY_NOTICES.md`.

## Changes made by Acoustic Agent

Acoustic Agent did not use a VLM to invent absorption coefficients. The source
coefficients remain upstream data. The project used VLM-assisted classification
and deterministic post-processing to:

1. map material names and descriptions to simulator-facing semantic objects;
2. organize 20 semantic classes into 16 compatible material families;
3. normalize the runtime representation to six octave bands;
4. derive four absorption classes from the 500 Hz, 1 kHz, and 2 kHz mean;
5. retain source-group identifiers and add confidence and quality flags; and
6. support seeded semantic-to-material sampling.

The mappings and normalized database are modified works and are not endorsed by
the upstream publishers.

## Public-release policy

Do not publish the complete database as open data until redistribution rights
for the ODEON/manufacturer and Acoustic Supplies subsets are documented in
writing. A public release should either obtain those permissions or build and
ship a reviewed redistributable subset.

This notice records the project's provenance review; it is not legal advice.
