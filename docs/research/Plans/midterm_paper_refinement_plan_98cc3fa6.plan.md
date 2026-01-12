---
name: Midterm Paper Refinement Plan
overview: "Refine the midterm paper to accurately reflect current progress: data pipeline completion (Steps 1-4), same-mission/filter matching discovery, scaling from 8 to 223 SNe, and differencing results. Structure for reuse in final paper."
todos: []
---

# Midterm Paper Refinem

ent Plan

## Overview

Refine `docs/research/astrid-midterm.md` to accurately reflect:

- **Current focus**: Data pipeline completion (Steps 1-4), not ML classification yet

- **Key discovery**: Same-mission and same-filter requirements for valid differencing

- **Progress**: 8 pilot SNe tested → scaling to 223 viable pairs

- **Results**: Successful differencing pipeline with quantitative metrics

## Structure Changes

### Section Reorganization

1. **Keep Sections 1-3** (Introduction, Background, Overview) - minor edits only
2. **Replace Section 4** - Shift from "star detection dataset" to "supernova transient detection pipeline"

3. **Remove/Consolidate Section 5** - U-Net star detection becomes brief mention in introduction

4. **Replace Section 6** - Results focus on differencing pipeline, not star detection

5. **Expand Section 7** - Challenges emphasize data constraints discovery

6. **Rewrite Section 8** - Current supernova work becomes main focus

7. **Update Section 9** - Next steps acknowledge ML classification as future phase

8. **Revise Section 10** - Conclusion reflects data pipeline achievements

## Key Data Points to Incorporate

### Data Acquisition Statistics

- **Catalog**: 6,542 known supernovae compiled from Open Supernova Catalog

- **Query results**: 1,110 SNe queried (from terminal: 42.4% viable = 471 SNe)

- **Current download**: 223 same-mission, same-filter pairs (all viable)

- **Storage**: 210GB raw → ~105GB after filtering to FITS only (from cleanup script)
- **File counts**: 27,427 files scanned → 10,940 FITS kept → 16,487 auxiliary files removed

### Pipeline Results (8 Pilot SNe)

- **SN 2014J**: 2120σ max significance, 100% overlap, 73 detections

- **Other SNe**: 412-797σ significance, 93-100% overlap, 73-629 detections

- **Same-mission requirement**: Only 8/19 (42%) had usable pairs in pilot

- **Filter matching**: Critical within same mission (e.g., UUU-UUU required)

## Section-by-Section Refinement Plan

### Section 1: Introduction

**Changes:**

- Clarify pivot from star detection to transient detection
- Emphasize data pipeline as current focus

- Set expectation: ML classification is next phase

**Key additions:**

- Brief mention of previous U-Net star detection work

- Statement that current work focuses on building training dataset for supernova detection

### Section 2: Background and Motivation

**Keep mostly as-is** - already well-written

**Minor addition:**

- Add note about scale: modern surveys generate millions of alerts per night

### Section 3: Project Overview and System Design

**Major rewrite:**

- Replace generic pipeline with 5-step supernova detection pipeline:

1. Data Acquisition (MAST queries, filtering)
2. Image Alignment (WCS reprojection)

3. Difference Imaging (ZOGY-style)

4. Candidate Generation (source detection - planned)

5. Real/Bogus Classification (CNN - future)

**Add visual pipeline diagram** (ASCII or reference to image)

### Section 4: Dataset Creation and Data Pipeline Development

**Complete rewrite** - Replace star detection content with:

#### 4.1 Supernova Catalog Compilation

- Compiled 6,542 known supernovae from Open Supernova Catalog

- RA/Dec coordinates, discovery dates, types

- Foundation for querying archival observations

#### 4.2 MAST Archive Querying

- Query MAST for observations before/after discovery dates

- Time windows: 3 years before, 2 years after discovery

- Multiple missions: SWIFT, GALEX, PS1, TESS, HST, JWST

#### 4.3 Critical Discovery: Same-Mission Requirement

- **Key finding**: Only 42.4% of queried SNe have viable pairs
- Cross-mission differencing fails (different PSFs, pixel scales, noise)

- Must query specifically for same-mission temporal coverage

#### 4.4 Filter Matching Within Missions

- Even same-mission requires matching filters

- Example: SWIFT UVW1 reference + UVM2 science = invalid

- Solution: Group by filter, only pair matching filters

#### 4.5 Data Volume and Management

- 223 viable pairs downloading (all same-mission, same-filter)

- Raw downloads: 210GB including auxiliary files

- Filtered to FITS: ~105GB (50% reduction)

- File management: Automated cleanup of non-FITS files

### Section 5: Image Differencing Pipeline Implementation

**New section** - Technical details:

#### 5.1 WCS Alignment

- `reproject.reproject_interp()` for pixel-perfect alignment

- Results: 93-100% overlap achieved
- Critical: Sub-pixel misalignment creates dipole artifacts

#### 5.2 Background Subtraction

- `photutils.Background2D` with median estimator

- Local background estimation (50-64 pixel boxes)
- Essential for transient detection

#### 5.3 PSF Estimation and Matching

- Estimate FWHM from bright stars using `DAOStarFinder`
- Typical SWIFT UVOT: ~3.5-4.0 pixels

- Convolve sharper image to match broader PSF

- Eliminates bright-star residuals

#### 5.4 Flux Normalization

- Robust median matching in overlap region

- Sigma-clipped statistics to ignore outliers

- Scale factor and offset computation

#### 5.5 Difference Image and Significance Map

- ZOGY-style optimal differencing

- Significance map: `SNR = diff / sqrt(noise²)`

- Uniform noise properties enable consistent thresholding

### Section 6: Results from Pilot Dataset

**Rewrite** - Focus on differencing results:

#### 6.1 Pipeline Validation on 8 Supernovae

- Successfully processed 8 same-mission pairs
- All achieved 93-100% overlap

- SN 2014J: 2120σ significance (brightest SN in decades)

- Other SNe: 412-797σ significance

#### 6.2 Detection Statistics

- Candidate detections: 73-629 per image
- 5σ threshold applied
- Known SN positions successfully marked

#### 6.3 Data Quality Observations

- Archive data varies significantly

- Some SNe have excellent coverage, others none

- Filter information sometimes missing from filenames

- FITS headers contain critical metadata

### Section 7: Challenges and Lessons Learned

**Expand** - Emphasize data constraints:

#### 7.1 Data Availability Constraints

- Same-mission pairs are rare (42.4% viable rate)

- Must query archives specifically for temporal overlap

- Filter matching further constrains viable pairs

#### 7.2 Technical Challenges

- Simple subtraction inadequate without PSF matching

- Alignment must be sub-pixel accurate

- Background subtraction essential

- Significance maps more useful than raw differences

#### 7.3 Data Management Challenges

- Large data volumes (210GB raw downloads)

- Need for automated filtering and cleanup

- FITS file format variations (extensions, 3D data)
- Archive data quality varies

#### 7.4 Iterative Refinement Process

- Started with 8 pilot SNe to validate pipeline

- Discovered same-mission requirement
- Scaled to 223 viable pairs based on learnings

- Research is iterative, not linear

### Section 8: Current Status and Scaling

**Rewrite** - Current work:

#### 8.1 Data Acquisition Progress

- 223 same-mission, same-filter pairs identified

- Currently downloading (50% complete after 24 hours)

- All pairs pre-validated as viable

#### 8.2 Pipeline Automation

- `SNDifferencingPipeline` class implemented

- Batch processing scripts ready

- Automation for alignment and differencing

#### 8.3 Next Phase: Training Data Generation

- Generate image triplets (science, reference, difference)

- Extract 63×63 pixel cutouts at detection positions

- Label using known SN coordinates (ground truth)
- Balance dataset (real vs bogus examples)

### Section 9: Next Steps and Future Direction

**Update** - Acknowledge ML as future:

#### 9.1 Immediate Next Steps

- Complete download of 223 SNe
- Run differencing pipeline on all pairs
- Generate training triplets

- Create labeled dataset

#### 9.2 Machine Learning Classification (Future Phase)

- CNN architecture for real/bogus classification

- 3-channel input: science, reference, difference triplets
- Training with augmentation

- Evaluation with AUCPR, precision, recall (not accuracy)

#### 9.3 Long-term Goals

- Production inference pipeline
- Active learning loop

- Cross-survey generalization

### Section 10: Conclusion

**Revise** - Reflect data pipeline achievements:

- Documented progress on data acquisition and differencing pipeline

- Discovered critical constraints (same-mission, filter matching)

- Successfully validated pipeline on 8 pilot SNe
- Scaling to 223 viable pairs for training dataset

- ML classification acknowledged as next phase

- Demonstrated feasibility of automated transient detection pipeline

## Quantitative Summary Table to Add

| Metric | Value |

|--------|-------|

| Total SNe in catalog | 6,542 |

| SNe queried | 1,110 |

| Viable pairs (42.4%) | 471 |

| Current download target | 223 |

| Pilot dataset tested | 8 |

| Best detection significance | 2120σ (SN 2014J) |

| Typical overlap achieved | 93-100% |

| Raw download size | 210GB |

| Filtered FITS size | ~105GB |

| Files scanned | 27,427 |

| FITS files kept | 10,940 |

## Files to Reference

- `docs/research/MIDTERM_NOTES.md` - Detailed technical notes

- `docs/research/SUPERNOVA_DETECTION_PIPELINE_GUIDE.md` - Pipeline architecture

- `notebooks/supernova_training_pipeline.ipynb` - Implementation details

- `src/differencing.py` - Pipeline code

- `scripts/identify_same_mission_pairs.py` - Data filtering logic

- Terminal outputs - Data statistics

## Writing Style Guidelines

- **Honest about progress**: Acknowledge what's done vs planned

- **Technical but accessible**: Explain concepts without oversimplifying

- **Data-driven**: Use specific numbers and metrics

- **Reflective**: Show learning process, not just results

- **Forward-looking**: Connect current work to future ML phase

## Implementation Approach

1. **Section-by-section refinement**: Work through each section sequentially

2. **Incorporate data points**: Add statistics from terminal outputs

3. **Maintain academic tone**: Professional but personal (first-person OK for self-study)

4. **Visual elements**: Consider adding pipeline diagram, results table