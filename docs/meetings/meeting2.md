# **AstrID** – *Meeting 2*

## Objective
Align on the next steps for an end-to-end pilot and agree on data sources, simple success measures, and short-term milestones.

## Recap
- Goal: Detect astronomical anomalies (transients) from imaging time-series.
- Scope (this phase): Small pilot across a few sky surveys to validate the pipeline and basic accuracy before scaling up.

## Data Sources (Plan)
- Mikulski Archive for Space Telescopes (MAST)
  - Focus: Pan-STARRS1 for deep reference images; Hubble for occasional deep context; TESS mainly for time information if useful.
  - Access: Programmatic queries (astroquery / MAST API). Be mindful of rate limits and any data-use rules.
  - Action: Create a short list of sky regions and targets, define a minimal set of fields to retrieve, and pull a small sample.

- NASA SkyView (on-demand image cutouts)
  - Surveys to try first: DSS2, SDSS, GALEX, 2MASS, WISE.
  - Standardize a default image size and pixel scale so model inputs are consistent. Define an ordered fallback if a survey is unavailable.
  - Action: Specify default bands and cutout size, then fetch a small set for testing.

- VizieR (catalog cross-matching)
  - Catalogs: Gaia DR3, SDSS DR16, Pan-STARRS DR2, AllWISE.
  - Use: Match positions to get labels/metadata (e.g., known sources, brightness) and simple features.
  - Action: Write basic queries with a small set of columns to keep downloads light.

- Zwicky Transient Facility (ZTF) or similar
  - Option: Public data releases and small image stamps for time-variable sources.
  - Action: Identify accessible samples to benchmark transient detection on real events.

- Selection Criteria
  - Choose a few RA/Dec tiles away from the crowded Galactic plane; set basic brightness limits; note seeing/point-spread differences.
  - Follow archive acknowledgment and data-use guidelines.

## Pipeline Milestones (Near Term)
1. Ingestion pilot: fetch a small number of cutouts per target from SkyView and Pan-STARRS via MAST.
2. Preprocessing defaults: basic alignment, background subtraction, and making image scales consistent.
3. Image differencing: run the planned subtraction method on the pilot set and record what settings work best.
4. Baselines:
   - Main baseline: U-Net style model for highlighting likely anomaly pixels/regions.
   - Simple comparisons: methods like Isolation Forest or One-Class SVM to check if the deep model adds value.
5. Experiment tracking: organize runs with clear names, parameters, and saved outputs so results are easy to compare.
6. Orchestration: a simple automated flow that runs ingest → preprocess → subtract → detect with minimal scheduling.

## Evaluation Plan (Plain Language Metrics)
- Precision ("how many flagged are real"): Of the candidates we mark as anomalies, what fraction are actually real events?
- Recall ("how many real we find"): Of the real events present, what fraction do we successfully detect?
- Localization quality ("how well we mark it"): When we highlight a region, how closely does it line up with the true location (e.g., overlap score/IoU, reported simply as a percent match)?
- False alarms per image ("noise level"): Average number of spurious detections per image. Keep this low so review time stays reasonable.
- Throughput ("speed per image"): Rough processing time per image on our current hardware, to understand scale-up needs.

Reporting style: For each dataset, present a small table with precision, recall, localization match, false alarms per image, and average processing time. Include a short note on threshold choices so results are comparable.

## Risks / Open Questions
- Ground truth scarcity: how to balance synthetic injections with a small set of known real events.
- Differences across surveys: bandpass, depth, and image sharpness can change detection quality.
- Access limits and rate limits: may need caching and staged downloads.
- Path to larger scale: rough estimates of compute and storage for the next phase.

## Timeline to Meeting 3 (~2 weeks)
- Week 1: finalize survey list; run ingestion pilot; agree on preprocessing defaults.
- Week 2: run image differencing on the pilot set; train the baseline model with synthetic injections; gather first-round metrics.
- Deliverables: sample cutouts, a few example differenced images, a simple metrics table, and an updated questions list.

## Requests for Advisor
- Confirm survey and catalog priorities for the pilot.
- Feedback on the plain-language metrics above and acceptable target ranges.
- Guidance on science priorities (e.g., supernovae vs. cataclysmic variables; preferred sky regions).
- Any help with data access or introductions to alert/broker resources if needed.

