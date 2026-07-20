# Acoustic Materials DB

The runtime source of truth is `acoustic_materials_v3.sqlite3`. It is the only
material data file loaded by a normal Acoustic Agent installation.

- 20 VLM semantic object classes
- 16 material families
- 3,741 acoustic material records from five source groups
- four mid-band absorption classes
- six absorption bands: 125, 250, 500, 1000, 2000, and 4000 Hz

Selection follows:

```text
semantic object -> compatible material family -> absorption class -> material instance -> six-band table
```

`MaterialLibrary.load()` caches the parsed runtime index for the process.
Sampling is deterministic for a semantic, class, optional material family, and
seed.

The complete SQLite database is intentionally bundled with the open-source
engine so an installed package can reproduce semantic material selection without
an external service. The VLM-assisted layer maps source material names and
descriptions to compatible simulator semantics; it does not generate acoustic
coefficients.

Apache-2.0 covers the project-authored engine, schema, taxonomy, mappings, QA
metadata, builder, and sampler. The bundled records retain source-specific
terms, and their inclusion does not create a new blanket license. Source counts,
URLs, citations, and machine-readable status are in `sources.json`. Read
`DATA_LICENSE.md` before redistribution. The offline database-authoring workspace
and rebuild tooling are intentionally not part of the runtime repository; they
remain available on the `archive/material-db-rebuild-v3` branch.
