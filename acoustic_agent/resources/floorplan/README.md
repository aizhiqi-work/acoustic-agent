# Floorplan V1

`floorplan_v1.sqlite3` is Acoustic Agent's runtime floor-plan scene resource. It
contains 15,376 audited residential scenes converted from the ResPlan dataset
into geometry that can be consumed directly by the acoustic solver.

The runtime transformation includes:

- filtering malformed, overlapping, disconnected, multilevel, and excessively
  complex records;
- normalizing coordinates to metric scale;
- cleaning and deduplicating room polygons;
- extruding walls, floors, ceilings, doors, and windows into acoustic surfaces;
- converting verified interior openings into portals;
- retaining room connectivity for same-room and cross-room simulation; and
- storing compact scene payloads as zlib-compressed JSON in SQLite.

The public Acoustic Agent feature is named **Floorplan**. **ResPlan** remains the
name of the upstream dataset and must be retained in attribution and citations.

The adapted resource is distributed under CC BY-NC-SA 4.0. Read
`DATA_LICENSE.md` before use or redistribution. Source and citation metadata are
also available in `source.json`.
