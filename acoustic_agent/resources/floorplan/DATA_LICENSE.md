# Floorplan V1 Data Terms

The Floorplan V1 SQLite resource is adapted from the ResPlan dataset by Mohamed
Abouagour and Eleftherios Garyfallidis.

- Dataset: ResPlan: A Large-Scale Vector-Graph Dataset of 17,000 Residential
  Floor Plans
- Dataset page: https://www.kaggle.com/datasets/resplan/resplan
- Paper: https://arxiv.org/abs/2508.14006
- DOI: https://doi.org/10.48550/arXiv.2508.14006
- Upstream dataset license: Creative Commons Attribution-NonCommercial-
  ShareAlike 4.0 International (CC BY-NC-SA 4.0)
- License: https://creativecommons.org/licenses/by-nc-sa/4.0/

## Acoustic Agent changes

Acoustic Agent filtered 17,107 source records to 15,376 eligible single-floor
scenes, normalized geometry to metric scale, cleaned room polygons, reconstructed
doors/windows/open connections, extruded 2D plans into acoustic surfaces and
portals, and stored the transformed scenes as zlib-compressed JSON in SQLite.
The resulting resource is intended for indoor acoustic simulation and is not
endorsed by the ResPlan authors.

## Requirements

Use and redistribution of this adapted resource must comply with CC BY-NC-SA
4.0. In particular, users must provide attribution, identify the changes, use
the resource only for noncommercial purposes, and distribute adaptations under
the same license. The Apache-2.0 project license applies to Acoustic Agent code;
it does not replace the license of this dataset-derived resource.

No warranties are given. This notice is not legal advice.
