# Plan Enhancements Informed by Recent Literature

This document tracks adjustments to the AstrID plan based on relevant articles and summaries, focusing on clear metrics, low acronym use, and practical next steps.

## Sources
- Supernova search with active learning in ZTF DR3 — A&A 672, A111 (2023)
  - https://www.aanda.org/articles/aa/full_html/2023/04/aa45172-22/aa45172-22.html
- Unsupervised anomaly detection in physics time-series (methodological insights) — Phys. Rev. D 103, 063011 (2021)
  - https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.063011
- Overview of anomaly detection approaches in astronomy (conceptual guide) — Astrobites (2024-07-07)
  - https://astrobites.org/2024/07/07/guest-anomaly-detection/

## Key Improvements to Incorporate

### 1) Human-in-the-loop active learning (ZTF/SNAD)
- Add a weekly loop: rank top outliers → brief human review → retrain using new labels to prioritize informative candidates.
- Track benefits:
  - Label efficiency: number of true events per 100 reviewed.
  - Pool yield: new high-quality candidates discovered per week.
  - Time-to-detection: hours from data arrival to surfacing a candidate.
- Implementation note: Integrate a review-and-retrain step into the workflow scheduler after each ingestion batch.

### 2) Plain-language evaluation metrics
- Keep simple and comparable measures:
  - Precision: of the flagged candidates, how many are real?
  - Recall: of the real events present, how many did we find?
  - Localization quality: how closely the highlighted region matches the true position (overlap percent reported simply).
  - False alarms per image: average number of spurious detections per image.
  - Throughput: average processing time per image on current hardware.
- Reporting: a small table per dataset with the five values above and a short note on the chosen decision threshold.

### 3) Robust data splitting and leakage control
- Split by sky region and observing night (not random images) to reduce scene leakage.
- Maintain disjoint regions and nights across train/validation/test.

### 4) Threshold calibration and “uncertain” bucket
- Calibrate anomaly thresholds on a small labeled validation set (e.g., isotonic or Platt-style mapping).
- Add an “uncertain” bucket to lower false alarms when confidence is low.

### 5) Combine unsupervised and supervised signals
- Start with an unsupervised score (e.g., reconstruction error or density-based rank) to find outliers.
- Train a lightweight classifier on reviewed labels to refine rankings.
- Use a blended score to order candidates for review.

### 6) Synthetic injections for grounded recall
- Inject a range of transient shapes and brightness levels into real backgrounds.
- Vary point-spread (seeing) and host background levels.
- Report recall as a function of brightness and size.

### 7) Reviewer interface and speed
- Provide consistent image stamps: science, reference, and difference side-by-side, with minimal metadata (position, score, nearest catalog match).
- Track reviewer time per 100 candidates to monitor burden.

### 8) Cross-matching and novelty measurement
- Nightly cross-match with catalogs (Gaia, Pan-STARRS, SDSS, AllWISE) and any public transient samples to tag known sources.
- Report “discovery lift”: fraction of found events not present in baseline catalogs/brokers.

### 9) Robustness checks and distribution shift
- Evaluate performance across different seeing conditions, crowding, and filters.
- Report metric changes under these conditions to show stability.

## Near-Term Actions (next 2 weeks)
- Define a small pilot set (20–30 images) from SkyView and Pan-STARRS for testing.
- Implement grouped splits by region and night.
- Stand up a simple review tool exporting ranked candidates with stamp triplets and CSV.
- Add calibration and an “uncertain” bucket to reduce false alarms.
- Run first synthetic injection sweep and report recall vs. brightness.
- Add a weekly active-learning meeting: review top-K, update labels, retrain.

## Advisor-Facing Deliverables for Next Meeting
- Ranked outlier list from pilot, with reviewer outcomes.
- One-page metrics table with precision, recall, localization match, false alarms/image, and time per image.
- A short demo of the active learning loop (how review feeds back into the model).
- A stratified slice (e.g., good vs. poor seeing) showing robustness.

## Notes and Citations
- The active learning loop is inspired by SNAD’s ZTF work, which showed that iterative human–machine review can surface missed events and improve efficiency. See: https://www.aanda.org/articles/aa/full_html/2023/04/aa45172-22/aa45172-22.html
- Threshold calibration and blended scoring reflect common practices in unsupervised anomaly search in physics and astronomy. See: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.063011 and the overview at https://astrobites.org/2024/07/07/guest-anomaly-detection/

