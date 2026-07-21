# Distributed FloorPlan Localization And Tracking

## Research question

Given a known residential FloorPlan, how many microphone channels and how many
spatially separated nodes are required to locate and track one active speaker
throughout the home, including across open doors?

The deployment must be source-position-independent: it may use room polygons,
semantic room labels, node coordinates, and portal connectivity, but it may
not inspect evaluation source positions or trajectories.

## Traditional baseline

No neural network is used. The pipeline consists of:

1. One ceiling-height candidate node at a safe interior point of every room.
2. Topology-aware minimax greedy node selection over a structural probe grid.
3. SRP-PHAT for local circular-array DOA.
4. RIR-onset TDOA for globally synchronized single microphones or array nodes.
5. A FloorPlan grid likelihood with portal-routed arrival bearings and travel
   times.
6. Robust DOA/TDOA likelihood fusion.
7. A constant-velocity Kalman filter for motion trend estimation.

For an array node, the placement objective accumulates bearing information
where the source and node share a room and discounted portal evidence
elsewhere. For synchronized single microphones, the objective uses a Fisher
information matrix built from portal-routed travel-time gradients. Emission
time is treated as a nuisance variable and removed with a Schur complement.
Greedy selection maximizes a lower-tail information quantile plus mean
coverage, favoring difficult regions without using any evaluation target.

## Train and test protocol

- Training FloorPlans: `0, 2, 7, 13`.
- Test FloorPlans: `20, 29, 41, 60`.
- The sets are disjoint.
- Training scenes select the minimax risk quantile from `0.05, 0.20, 0.35`.
- Test source positions are sampled only after deployment is frozen.
- Every room contributes one static test point.
- Every test FloorPlan contributes one six-frame trajectory crossing an open
  interior door.

The static test currently contains 35 source positions. The motion test
contains four trajectories and 24 frames. All acoustic observations come from
Acoustic Agent RIRs at `preview` quality. The localization algorithm receives
the FloorPlan and sensor coordinates, but never the source ground truth.

## Compared deployments

The experiment separates physical node count from channel count:

- one 8-channel circular array;
- two, three, or four distributed 4-channel arrays, with and without
  inter-node synchronization;
- three, four, six, or eight synchronized single microphones;
- two 4-channel arrays plus four single microphones;
- six-single-microphone placement by largest rooms, farthest rooms, and the
  proposed topology-aware minimax strategy;
- six single microphones with repeated 100 and 500 microsecond clock-offset
  perturbations.

Run the study:

```bash
python -m research.doa.run_distributed
```

The four-FloorPlan study is a compact algorithm comparison. The statistically
broader scaling experiment is:

```bash
python -m research.doa.run_stratified
```

It uses ten unseen validation FloorPlans for every exact room count from 4 to
14, with one sample from every within-stratum area decile. Five separate
FloorPlans per room count calibrate the placement risk parameter. It tests each
integer synchronized-single-microphone budget from three microphones through
one microphone per room and bootstraps confidence intervals by FloorPlan.

Generated CSV, JSON, cache, and Markdown evidence is written under
`research/results/distributed-floorplan/` and
`research/results/distributed-floorplan-stratified/`; both are ignored by Git.
Compact report snapshots and aggregate CSV files are committed under
[`evidence/`](evidence/README.md).

## Evaluation

Static metrics:

- median and 90th-percentile 2D position error;
- room classification accuracy;
- error and room accuracy when no selected node is in the source room;
- physical nodes and total microphone channels.

Motion metrics:

- raw and Kalman-filtered position error;
- start-to-end motion-direction error;
- room accuracy;
- open-door transition detection delay.

The automatic selector defines a configuration as adequate when median error
is at most 1.0 m, P90 error is at most 2.0 m, and room accuracy is at least
85%. It then selects the adequate configuration with the fewest channels.

## Current result

The stratified experiment scans all `15,376` compiled FloorPlans and retains
connected, geometrically valid layouts. Room counts 4 through 14 each use five
calibration and ten unseen validation layouts, with validation sampling across
area deciles. This gives 110 validation FloorPlans, 990 source positions, and
8,030 microphone-budget localization cases.

The minimum adequate synchronized-single deployments are:

| Rooms | Microphones | Median | P90 | Room accuracy |
|---:|---:|---:|---:|---:|
| 4-6 | 4 | 0.61-0.92 m | 1.41-1.87 m | 90.0-100% |
| 7-8 | 5 | 0.62-0.63 m | 1.43-1.88 m | 96.2-98.6% |
| 9-10 | 6 | 0.61-0.62 m | 1.55 m | 94.4-98.0% |
| 11-14 | 7 | 0.52-0.63 m | 1.50-1.82 m | 86.9-98.2% |

The observed compact rule is
`min(7, max(4, ceil(room_count / 2) + 1))`. It is a validation-set summary,
not a universal hardware law. At each room count's recommended budget, the
within-stratum small, medium, and large area groups produce median errors of
`0.55 m`, `0.63 m`, and `0.74 m`; their P90 errors are `1.44 m`, `1.60 m`, and
`1.82 m`. Homes at least 150 m2 are the most difficult absolute-area group at
`0.71 m` median, `1.92 m` P90, and `92.1%` room accuracy.

The broader validation supersedes the four-layout split for microphone-count
scaling. The smaller study remains useful for comparing sensor families and
clock perturbations:

On the fixed four-FloorPlan test split, the minimum adequate configuration is
six globally synchronized single microphones using topology-aware placement:

- median position error: `0.63 m`;
- P90 position error: `1.69 m`;
- room accuracy: `94.3%`;
- cross-room-only median error: `1.11 m`;
- cross-room-only room accuracy: `81.8%`.

Eight synchronized single microphones improve the result to `0.41 m` median,
`1.33 m` P90, and `100%` room accuracy. Six microphones placed only in the
largest rooms produce `1.15 m` median error and `74.3%` room accuracy. Farthest
room placement produces `0.77 m` and `82.9%`, showing that topology-aware
lower-tail coverage is important.

DOA-only asynchronous arrays perform poorly across rooms because a remote
array often observes a portal or reflection direction rather than the direct
bearing to the speaker. Globally synchronized array nodes improve strongly by
adding inter-node TDOA, but spend more channels at fewer spatial locations.
This experiment therefore supports spatial distribution over dense local
sampling for whole-home single-speaker positioning.

For the four door-crossing trajectories, six microphones plus a
constant-velocity Kalman filter achieve `0.94 m` filtered median error, `6.3°`
median trend-direction error, `95.8%` room accuracy, and zero-frame median
absolute portal-transition delay.

These numbers are conclusions for this simulation protocol, not universal
hardware requirements.

## Related research

- Hahmann et al., [Sound source localization using multiple ad hoc distributed
  microphone arrays](https://doi.org/10.1121/10.0011810), JASA Express Letters,
  2022.
- Liu et al., [Deep Learning Based Stage-wise Two-dimensional Speaker
  Localization with Large Ad-hoc Microphone Arrays](https://arxiv.org/abs/2210.10265),
  2022.
- An et al., [Reflection-Aware Sound Source Localization](https://arxiv.org/abs/1711.07791),
  2017.
- Evers et al., [The LOCATA Challenge](https://arxiv.org/abs/1909.01008), 2019.
- Ettinger and Freund, [Particle Filtering on the Audio Localization
  Manifold](https://arxiv.org/abs/1003.0659), 2010.

## Limitations and next experiments

- Single microphones are assumed globally synchronized with known positions.
- Clock offsets are fixed perturbations, not a complete drifting-clock model.
- Only one active source is present.
- Source activity detection and speech intermittency are not yet modeled.
- FloorPlan geometry and portal states are assumed correct.
- Simulation-to-real validation is still required.
- Privacy-constrained placement should exclude sensitive rooms and be reported
  as a separate deployment condition.
- Multiple speakers require data association before extending the tracker.
