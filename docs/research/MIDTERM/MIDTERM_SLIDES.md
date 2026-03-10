# AstrID Midterm Presentation — Slide Deck Plan

**Duration**: ~20 minutes (25 slides, ~45–50 seconds per slide on average)  
**Audience**: Faculty advisors (Prof. Meneses, Prof. Arkos) and peers  
**Format**: Title + bullet points + speaker notes per slide  

---

## Slide 1 — Title Slide
**Title**: AstrID: Supernova Detection Pipeline  
**Subtitle**: Machine Learning for Astronomical Transient Identification  
**Details**:
- Chris Lawrence
- CSCI 491 — Senior Research Project
- Midterm Presentation — February 2026
- Advisors: Professor Luis Meneses, Professor Gregory Arkos

**Speaker Notes**: Brief introduction — name, course, project title, and advisors.

---

## Slide 2 — Agenda / Roadmap
**Title**: What We'll Cover Today
**Bullets**:
- The problem: scale of modern astronomical surveys
- AstrID's pipeline architecture (5 stages)
- Key discovery: same-mission requirement
- Image differencing technique and results
- Current dataset and proof-of-concept triplets
- Lessons learned and next steps

**Speaker Notes**: Give the audience a map of the talk so they know what to expect. ~20 min, save questions for the end or ask as we go.

---

## Slide 3 — The Problem: Scale of Modern Astronomy
**Title**: Why Automation Matters
**Bullets**:
- Modern surveys (e.g., Rubin/LSST) will generate ~10 million alerts per night
- Far too much data for human inspection
- Transient events (supernovae, novae, variable stars) appear briefly and fade
- Missing a supernova = losing a unique scientific opportunity
- Automation is not a convenience — it is a necessity

**Speaker Notes**: Emphasize the sheer scale. LSST alone will produce 10M alerts/night. Humans can't keep up — we need automated systems that can flag real events and filter out noise.

---

## Slide 4 — What Are Supernovae?
**Title**: Supernovae: Exploding Stars
**Bullets**:
- Catastrophic explosions at the end of a star's life
- Briefly outshine their entire host galaxy (billions of stars)
- Critical for cosmology: Type Ia SNe are "standard candles" for measuring cosmic distances
- Extremely rare: ~1 per century per galaxy
- Detection requires comparing images taken at different times

**Speaker Notes**: Supernovae are some of the most energetic events in the universe. They're scientifically invaluable but rare and fleeting — which is exactly why we need automated detection.

---

## Slide 5 — The "Bogus" Problem
**Title**: The False Positive Challenge
**Bullets**:
- Classical detection algorithms flag 73–629 candidates per image at 5σ threshold
- Only ~1 of those is a real supernova
- The rest: cosmic rays, subtraction artifacts, variable stars, galaxy nuclei
- A model predicting "always bogus" gets 99.99% accuracy — but is scientifically useless
- Need ML to separate real transients from artifacts

**Speaker Notes**: This is the core motivation for ML. Classical algorithms do the physics, but they can't distinguish real from bogus on their own. A classifier that always says "no supernova" would be highly accurate by metrics but scientifically worthless.

---

## Slide 6 — What Is AstrID?
**Title**: AstrID: Project Overview
**Bullets**:
- End-to-end pipeline for automated supernova detection
- Combines classical astronomy techniques with machine learning
- Handles the full workflow: data acquisition → preprocessing → image comparison → detection → classification
- Built from scratch — not using existing survey pipelines
- Emphasis on hands-on experimentation and system design

**Speaker Notes**: AstrID is designed to handle the entire problem — from downloading raw telescope data all the way to classifying detections as real or bogus. This project is about building the whole thing myself, understanding each component deeply.

---

## Slide 7 — Pipeline Architecture (High-Level)
**Title**: AstrID Pipeline: 5 Stages
**Bullets**:
- **Stage 1 — Query**: Retrieve observation metadata from MAST archive
- **Stage 2 — Filter**: Ensure same-mission compatibility between reference and science images
- **Stage 3 — Download**: Acquire FITS files with smart filtering (60–80% size reduction)
- **Stage 4 — Organize**: Create training-ready directory structures
- **Stage 5 — Differencing**: Full astronomical image differencing pipeline
- Output: difference images + significance maps + detection masks

**Speaker Notes**: Walk through each stage briefly. The key insight is that each stage has a specific job, and the modular design means we can test, debug, and improve each independently.

---

## Slide 8 — Data Acquisition: The Supernova Catalog
**Title**: Starting with Known Supernovae
**Bullets**:
- Compiled catalog of 6,542 known supernovae with coordinates and discovery dates
- Query the MAST (Mikulski Archive for Space Telescopes) archive
- Download **reference images** (before SN appeared) and **science images** (after discovery)
- Missions available: SWIFT, GALEX, PS1 (Pan-STARRS)
- Temporal windowing: reference = pre-discovery, science = post-discovery

**Speaker Notes**: We start with known supernovae so we have ground truth for training. MAST is NASA's main archive for space telescope data. We query for images taken before and after each supernova was discovered.

---

## Slide 9 — FITS Files: Astronomical Image Data
**Title**: Working with FITS Files
**Bullets**:
- FITS (Flexible Image Transport System) — standard format for astronomical images
- Contains pixel data + rich metadata (WCS coordinates, exposure info, filters)
- Not simple JPEGs — extensions, HDUs, 3D data arrays, compressed variants (.fits.gz)
- WCS (World Coordinate System) maps pixels to sky coordinates (RA/Dec)
- Handling compressed FITS led to a **296% increase** in available training pairs (56 → 222)

**Speaker Notes**: FITS files are the bread and butter of astronomy data. They're complex — multiple extensions, 3D arrays, compressed formats. Discovering that we were missing compressed files was a major breakthrough that nearly tripled our dataset.

---

## Slide 10 — Pilot Study: 19 Supernovae
**Title**: Pilot Study Results
**Bullets**:
- Started with 19 known supernovae from the catalog
- Downloaded reference + science images from MAST
- Attempted image differencing on all pairs
- Only **8 of 19 (42%)** had usable same-mission pairs
- 11 had cross-mission data (e.g., PS1 reference + SWIFT science)
- Cross-mission differencing produced massive, unusable artifacts

**Speaker Notes**: The pilot study was crucial. It looked like we had plenty of data, but when we actually tried to use it, more than half was unusable. This led to our most important discovery.

---

## Slide 11 — Critical Discovery: Same-Mission Requirement
**Title**: Why Cross-Mission Differencing Fails
**Bullets**:
- Different telescopes have fundamentally different characteristics:
  - Point Spread Functions (PSFs)
  - Filter bandpasses
  - Pixel scales
  - Noise properties
- Subtracting images from different missions produces massive artifacts
- These artifacts completely swamp any real transient signal
- **Requirement**: Reference and science images must come from the same mission AND same filter

**Speaker Notes**: This was the single most important discovery of the project. It seems obvious in hindsight, but it fundamentally reshaped the entire pipeline design. Every telescope sees the sky differently — you can't compare apples and oranges.

---

## Slide 12 — Same-Mission Impact on Dataset
**Title**: Impact of the Same-Mission Constraint
**Bullets**:
- Initial approach: download everything, filter later → 75% wasted downloads
- New approach: filter at query stage for same-mission pairs before downloading
- Result: **60–80% reduction** in download size
- Success rate improved: **25% → 75–100%** for creating complete training pairs
- Current dataset: 222 complete pairs across SWIFT, GALEX, PS1
- Lesson: understanding your data constraints early saves enormous effort

**Speaker Notes**: Once we understood this constraint, we built it into the pipeline. Instead of downloading everything and throwing most of it away, we filter first. This was a huge efficiency gain.

---

## Slide 13 — Production Dataset Statistics
**Title**: Current Production Dataset
**Bullets**:
- **222 complete supernova pairs** (reference + science images)
- **2,282 FITS files** total (1,170 reference, 1,112 science)
- **Mission breakdown**:
  - GALEX: 105 pairs
  - SWIFT: 17+ pairs
  - PS1: additional pairs
- **99.1% download success rate** for SNe with same-mission observations
- 15 YAML config files for reproducible dataset generation

**Speaker Notes**: This is a solid foundation. 222 pairs gives us enough to start training a classifier. The high success rate shows the pipeline is reliable once we have compatible data.

---

## Slide 14 — Image Differencing: The Core Technique
**Title**: What Is Image Differencing?
**Bullets**:
- Classical astronomical technique: subtract a reference image from a science image
- Anything that changed between the two epochs appears as a residual signal
- Simple pixel-by-pixel subtraction is **inadequate** — produces massive artifacts
- Must carefully: align images, match PSFs, normalize flux, subtract backgrounds
- Done correctly: only real changes (transients) remain in the difference image

**Speaker Notes**: Image differencing is conceptually simple — subtract two images and look for what's new. But the devil is in the details. Without careful preprocessing, the difference image is dominated by artifacts, not real signals.

---

## Slide 15 — Differencing Pipeline: 9 Stages
**Title**: AstrID Differencing Pipeline
**Bullets**:
1. FITS File Loading (robust HDU handling)
2. WCS Alignment (sub-pixel reprojection)
3. Background Estimation & Subtraction
4. PSF Estimation (FWHM from bright stars)
5. PSF Matching (Gaussian convolution)
6. Flux Normalization (robust median matching)
7. Difference Image Computation
8. Significance Map Generation (SNR per pixel)
9. Source Detection (DAOStarFinder, 5σ threshold)

**Speaker Notes**: Walk through the nine stages. Each addresses a specific physical or instrumental challenge. Skip any one, and the final result is garbage. This is why data quality matters more than model complexity.

---

## Slide 16 — WCS Alignment
**Title**: Stage 2: World Coordinate System Alignment
**Bullets**:
- Problem: images taken at different times have different pointings and rotations
- Even **1-pixel misalignment** creates massive dipole residuals around every source
- Solution: reproject science image onto reference WCS grid using `reproject.reproject_interp()`
- Achieves sub-pixel alignment accuracy
- Result: **93.6%–100% overlap** on processed supernova pairs

**Speaker Notes**: This is arguably the most critical step. Sub-pixel accuracy is essential — even a tiny offset creates bright artifacts around every star that would mask real transients. We use the WCS metadata in FITS headers to align images precisely.

---

## Slide 16b — WCS Alignment: A Recent Fix
**Title**: WCS Misalignment — Reverting to a Simpler Approach
**Bullets**:
- Initially struggled with WCS misalignment: crosshairs in training triplets were off, and sigma values were wrong
- Reverted to a simpler, proven method from a previous project instead of the more complex pipeline logic
- Fix: one shared FITS loader and consistent use of the header’s WCS for both reference and science images, so sky→pixel conversion is reliable everywhere

**Speaker Notes**: When visualizations showed crosshairs and sigma values were off, the issue was inconsistent WCS handling. Going back to a simpler “WCS from header, then world_to_pixel” approach that had worked in another project resolved it — a good reminder that simpler, proven methods often beat clever ones.

---

## Slide 17 — PSF Matching & Background Subtraction
**Title**: Stages 3–5: Preparing Images for Subtraction
**Bullets**:
- **Background subtraction**: Remove varying sky levels using local median filtering (50–64 px boxes)
- **PSF estimation**: Measure FWHM from bright stars (typically 3.5–4.0 px for SWIFT UVOT)
- **PSF matching**: Convolve the sharper image with a Gaussian kernel to match the broader PSF
- Without PSF matching: bright stars leave large residuals in the difference image
- With PSF matching: residuals are dramatically reduced, revealing real transients

**Speaker Notes**: The PSF is the telescope's "fingerprint" — how it blurs point sources. Different observations have different seeing conditions, so PSFs differ. We match them so that when we subtract, bright stars cancel out cleanly.

---

## Slide 18 — Significance Maps & Source Detection
**Title**: Stages 8–9: Finding Transients
**Bullets**:
- **Significance map**: Each pixel = signal-to-noise ratio (difference / combined noise)
- Enables consistent thresholding across images with different noise properties
- "Whitened noise" — uniform statistical properties regardless of original image depth
- **Source detection**: DAOStarFinder on significance map at 5σ threshold
- Shape filters (sharpness bounds) reject cosmic rays and extended sources
- Output: list of candidate transient positions with significance values

**Speaker Notes**: Instead of thresholding raw pixel values (which vary between images), we create a significance map where every pixel tells us how many standard deviations above noise it is. A 5σ detection means there's a 1-in-3.5-million chance it's random noise.

---

## Slide 19 — Results: SN 2014J Detection
**Title**: Brightest Detection: SN 2014J
**Bullets**:
- SN 2014J: brightest supernova in decades (in galaxy M82)
- Detected at **2120σ significance** — unmistakable signal
- SWIFT UVOT U-band filter
- Found at pixel coordinates (447.9, 555.4) — matches known position
- 73 total candidate detections in the image; 1 is the supernova
- Demonstrates pipeline works correctly when inputs are good

**Speaker Notes**: SN 2014J is our best example. It's so bright that the detection is overwhelming — 2120 times above the noise level. This gives us confidence that the pipeline is working correctly. But notice: even for this obvious case, there are 72 other false detections.

---

## Slide 20 — Batch Processing Results
**Title**: Multi-Supernova Batch Results
**Bullets**:
- 5 supernovae fully processed: SN 2014J, 2014ai, 2014bh, 2014bi, 2014cs
- All SWIFT UVOT U-band (uuu filter)
- Maximum significance range: **412σ to 2120σ**
- Detection counts: **73–629 candidates** per difference image
- Image overlap: **66%–100%** after WCS alignment
- Consistent pipeline behavior across multiple targets

**Speaker Notes**: The pipeline is consistent — it finds real supernovae across multiple targets. The wide range in detection counts shows how variable the false positive problem is. Some images have 73 candidates, others have 629, but only 1 is real in each.

---

## Slide 21 — The False Positive Problem (Detailed)
**Title**: Why ML Classification Is Needed
**Bullets**:
- At 5σ threshold: 73–629 candidates, but only ~1 real per image
- False positive sources:
  - Cosmic rays (sharp, single-pixel spikes)
  - Subtraction artifacts (imperfect alignment residuals)
  - Variable stars (real variability, but not supernovae)
  - Host galaxy nuclei (AGN activity, not SNe)
- **Accuracy is the wrong metric**: "always predict bogus" = 99.99% accuracy
- Correct metrics: precision, recall, F1, **AUCPR** (area under precision-recall curve)

**Speaker Notes**: This is the bridge to the ML phase. Classical detection does the physics perfectly, but it can't tell a supernova from a cosmic ray. That's where a CNN comes in — pattern recognition at scale. And we must use the right metrics; accuracy is meaningless with this class imbalance.

---

## Slide 22 — Training Triplets: Proof of Concept
**Title**: Image Triplets for CNN Input
**Bullets**:
- **Triplet** = 63×63 pixel cutout of (science, reference, difference) images
- 3-channel input — analogous to RGB channels for a regular CNN
- Labels: **Real (1)** = centered on known SN position; **Bogus (0)** = artifacts / random
- Current proof of concept: **57 samples** (30 real, 27 bogus) with augmentation
- Augmentation: rotation, flipping for data variety
- Triplet pipeline is built and tested — needs to run at scale

**Speaker Notes**: This is how we bridge classical detection and ML. We cut out small stamps around each detection — the CNN sees three "channels" showing the sky before, after, and the difference. With 57 samples we've proven the pipeline works, but we need thousands for real training.

---

## Slide 23 — Configuration-Driven Reproducibility
**Title**: YAML Configs & Modular Design
**Bullets**:
- All pipeline runs driven by YAML configuration files
- 15 configs targeting different missions, filters, temporal windows
- Examples: `swift_uv_dataset.yaml`, `galex_golden_era.yaml`, `best_yield_combined.yaml`
- Two execution modes:
  - `run_pipeline_from_config.py` — full 5-stage batch run
  - `run_pipeline_per_sn.py` — one SN at a time, lower memory, checkpoint/resume
- Any result can be exactly reproduced by rerunning with the same config

**Speaker Notes**: Reproducibility is central to scientific work. Every dataset we generate can be reproduced by anyone with the same config file. The modular design also lets us test individual stages without running the whole pipeline.

---

## Slide 24 — Mission-Specific Training Strategy
**Title**: Leveraging Modularity for ML Training
**Bullets**:
- Config system enables sophisticated training strategies:
  1. Train separate models per mission (SWIFT, GALEX, PS1)
  2. Train a combined model on all missions for generalization
  3. Cross-mission validation: train on SWIFT+PS1, test on GALEX
  4. Filter-specific models for specialized wavelength ranges
- Goal: learn physics of transients, not instrument artifacts
- Domain shift is a fundamental challenge in astronomical ML

**Speaker Notes**: The modular configuration system isn't just for convenience — it enables real scientific experiments. By training on one telescope and testing on another, we can prove the model has learned actual supernova physics rather than just memorizing what artifacts look like on a specific instrument.

---

## Slide 25 — Methodology Alignment with Literature
**Title**: How AstrID Aligns with Published Research
**Bullets**:
- Based on published supernova detection pipelines (e.g., ZTF/Braai)
- **Implemented (Phases 1–4)**:
  - Data acquisition, calibration, differencing, source extraction
- **Remaining (Phases 5–8)**:
  - CNN classifier (Braai-style architecture)
  - Training with class imbalance handling
  - Deployment & real-time inference
  - Evaluation (AUCPR, precision-recall)
- Simplified ZOGY (Gaussian PSF matching vs. full Fourier-space subtraction)

**Speaker Notes**: We're following established methodology from the literature, particularly the ZTF Braai pipeline. We've completed the first four phases (data through detection) and the remaining four (ML through deployment) are the focus of the second half.

---

## Slide 26 — Lessons Learned
**Title**: Key Lessons from the First Half
**Bullets**:
- **Data quality > model complexity**: Better preprocessing beat fancier models every time
- **Discover constraints early**: Same-mission requirement reshaped the entire pipeline
- **Small details matter**: Compressed FITS support → 296% more training pairs
- **Resource constraints matter**: Differencing was killed by OOM on large PS1 images; addressed with float32, explicit cleanup, and batch processing
- **Invest in infrastructure**: Modular design + configs accelerated all later work
- **Research is non-linear**: Progress came from revisiting assumptions and iterating
- **Debugging ML ≠ debugging software**: Emergent behavior from data-parameter interactions
- **WCS alignment**: When crosshairs and sigma were wrong, reverting to a simpler WCS-from-header approach from a prior project fixed it
- **Pipeline progress/resume**: Long pipeline runtimes made iterating on WCS (and other fixes) painful. We added a per-dataset progress file (`pipeline_progress.json`) that records which SNe completed differencing; the next run skips those and only processes the remainder, so reruns are much faster

**Speaker Notes**: These lessons are as important as the technical results. The biggest gains came not from sophisticated algorithms but from understanding the data better. Infrastructure investment (configs, modularity) felt slow at first but paid off enormously.

---

## Slide 27 — Next Steps: Scale the Dataset
**Title**: Phase A: Scaling Difference Images
**Bullets**:
- Current: 5 fully processed SNe, 57 triplet samples
- Target: **200+ difference image sets** with quality filtering
- Enforce minimum **85% overlap** threshold
- Run differencing across all 222 pairs using existing configs
- Expand SN coordinate catalog for better real/bogus labeling
- Deliverable: Large-scale difference images + processing summaries

**Speaker Notes**: The immediate priority is scaling up. We have the pipeline — now we need to run it at scale. 200+ good difference images will provide enough data for meaningful ML training.

---

## Slide 28 — Next Steps: CNN Classifier
**Title**: Phases B–C: Triplets & CNN Architecture
**Bullets**:
- **Phase B**: Generate thousands of 63×63 triplets from scaled differencing output
  - Balanced classes (~50% real / 50% bogus)
  - Augmentation: rotation, flip, save NPZ format
- **Phase C**: Implement Braai-style CNN
  - Input: 63×63×3 (science, reference, difference)
  - Architecture: Conv2D(32)→Conv2D(64)→Conv2D(128)→Dense(256)→Sigmoid
  - Output: probability [0, 1] (bogus → real)
  - Framework: TensorFlow/Keras or PyTorch

**Speaker Notes**: Once we have enough difference images, we generate triplets and feed them to a CNN. The architecture is well-established in the literature — three conv blocks, a dense layer, and sigmoid output. The challenge is getting enough quality training data.

---

## Slide 29 — Next Steps: Training & Evaluation
**Title**: Phases D–E: Training, Evaluation & Deployment
**Bullets**:
- **Training**: Temporal/spatial splits to avoid data leakage
- **Class imbalance**: Weighted loss or oversampling — model must not collapse to "always bogus"
- **Metrics**: Precision, recall, F1, **AUCPR** — not accuracy
- **Threshold**: Prefer minimizing false negatives (missing a SN is catastrophic)
- **Stretch goals**:
  - Cross-mission validation (train on one telescope, test on another)
  - Real-time inference API (FastAPI)
  - Active learning loop for human-in-the-loop refinement

**Speaker Notes**: Evaluation is where the science meets the engineering. We need the right metrics (not accuracy), the right splits (no leakage), and the right threshold (permissive, because missing a supernova is worse than flagging a false positive for human review).

---

## Slide 30 — Project Timeline
**Title**: Remaining Timeline
**Bullets**:
- **Feb–Mar 2026**: Scale differencing to 200+ pairs; generate triplet dataset
- **Mar–Apr 2026**: Implement and train CNN classifier; evaluate with AUCPR
- **Apr 2026**: Cross-mission validation experiments
- **Apr–May 2026**: (Stretch) Inference API and active learning prototype
- **May 2026**: Final report and presentation
- Foundation is solid — ML phase is the focus going forward

**Speaker Notes**: We have a clear roadmap. The hardest part (understanding the data and building the pipeline) is done. The remaining work is primarily ML — building the classifier and evaluating it rigorously.

---

## Slide 31 — Conclusion
**Title**: Summary & Key Takeaways
**Bullets**:
- Built a **production-ready 5-stage pipeline** from scratch
- Discovered critical constraints (same-mission requirement) through experimentation
- Implemented a **9-stage differencing pipeline** that successfully detects supernovae (412σ–2120σ)
- Created **222 training pairs** (2,282 FITS files) across 3 missions
- Proved triplet generation works at small scale (57 samples)
- Pipeline is verified, modular, and ready for ML scaling
- **Key insight**: Data quality and infrastructure investment matter more than model complexity

**Speaker Notes**: The first half of the project has established a solid foundation. We understand the data, the constraints, and the pipeline works. The second half is about scaling up and applying ML to solve the false positive problem.

---

## Slide 32 — Questions
**Title**: Questions?
**Bullets**:
- Thank you!
- GitHub: [repository link]
- Contact: Chris Lawrence
- Happy to discuss technical details, next steps, or methodology

**Speaker Notes**: Open the floor for questions. Be prepared to discuss: same-mission discovery in depth, differencing pipeline details, why specific metrics matter, and timeline feasibility.
