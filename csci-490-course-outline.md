# CSCI 490 Senior Research Project Outline

**Student:** Chris Lawrence
**Project Title:** Astronomical Identification: Temporal Dataset Preparation and Anomaly Detection
**Supervisor:** Prof. Luis Meneses
**Committee members:** Prof. Gregory Arkos
**Proposal:** CSCI 490 Senior Research Project (3 credits)
**Duration:** September 2, 2025 through December 31, 2025

---

## Topic Outline

The AstrID project was developed as a system for identifying stars in astronomical images using FITS data and a U-Net machine learning model, developed for a CSCI 478 course in F24 semester. This existing system can successfully localize and classify stars with reasonable accuracy, serving as a foundation for further research.

This project will extend AstrID in one key area:

**Temporal Dataset Preparation and Anomaly Detection** – developing a comprehensive pipeline for processing time-series astronomical observations, cleaning and preparing temporal datasets, and implementing baseline anomaly detection algorithms to identify transient events (e.g., supernovae, brightness variations) across multiple observations.

This builds directly on the completed AstrID framework while focusing on the critical foundation work needed for temporal analysis.

---

## Project Description

The project will employ a systematic approach combining data engineering, machine learning research, and astronomical data analysis to establish a robust temporal dataset pipeline and baseline anomaly detection capabilities.

**Core Research Components:**

* **Temporal Dataset Preparation:** Investigation and implementation of data cleaning, preprocessing, and ingestion techniques for time-series astronomical imaging data. This includes developing robust pipelines for handling FITS format images across multiple observations, ensuring data quality, and creating standardized temporal datasets.

* **Baseline Anomaly Detection:** Research and implementation of fundamental anomaly detection algorithms for astronomical time-series data, with a focus on identifying supernovae and other transient anomalies. This work will extend the existing AstrID U-Net framework to support temporal analysis.

**Advanced Implementation Areas:**
* Temporal data ingestion and validation pipelines
* Data quality assessment and cleaning algorithms
* Time-series alignment and normalization techniques
* Baseline anomaly detection models extending AstrID's U-Net architecture
* Temporal analysis pipelines for multi-observation datasets
* Automated data ingestion and result logging systems

**Research Methodology:** The project will employ a mixed-methods approach combining:
* Literature review of astronomical time-series data processing and anomaly detection approaches
* Comparative analysis of existing temporal data preparation solutions in astronomy
* Quantitative evaluation with data quality metrics and anomaly detection accuracy
* Iterative development based on data pipeline reliability and detection performance
* Systematic testing of temporal data handling across different astronomical datasets

**Evaluation Framework:** The research will be evaluated through multiple criteria including:
* Technical functionality and reliability of temporal data processing pipeline
* Data quality metrics and consistency across time-series observations
* Performance of baseline anomaly detection compared to static identification approaches
* Scalability and generalizability of the temporal solution to other astronomical datasets

**Project Criteria for Different Grade Levels:**

* **A+ (Pass with honours):** Exceptional temporal data pipeline reliability (>95% data quality), high anomaly detection accuracy (>85%), comprehensive documentation, and significant contribution to temporal astronomical data processing methodology.

* **B+ (Pass with distinction):** High temporal data pipeline reliability (>90% data quality), good anomaly detection accuracy (>75%), thorough documentation, and clear contribution to temporal data processing.

* **C (Pass):** Working temporal data pipeline (>80% data quality), functional anomaly detection system (>65% accuracy), adequate documentation, and demonstrated learning outcomes.

* **F (Fail):** Incomplete or non-functional temporal pipeline, inadequate documentation, or failure to meet minimum project requirements.

**Future Research Directions:** The temporal dataset framework established in this project will enable future investigations into:
* Advanced anomaly detection algorithms for astronomical time-series
* Real-time transient event detection in astronomical surveys
* Integration with telescope observation scheduling systems
* Analytics for understanding temporal patterns in astronomical objects

---

## Project Deliverables and Schedule

The student will meet weekly with Prof. Meneses (schedule TBA) to review:
- Progress on milestone goals
- Technical challenges or architectural decisions
- Feedback on code, documentation, and presentation preparation

At each meeting, the student will provide a short (10-15 minute) briefing on the project state and bring current copies of relevant documents.

### Fall 2025 Project Schedule:

| Date | Deliverable |
|------|-------------|
| Sept 2-16 | Finalize proposal and registration form; project kickoff; environment setup |
| Mid Sept | Literature review on temporal astronomical data processing and anomaly detection; initial research log |
| Oct | Dataset preparation and cleaning pipeline development; temporal data validation framework |
| Nov | Implement baseline anomaly detection model; temporal data ingestion system; preliminary evaluation |
| Dec | Complete temporal pipeline; final evaluation and testing; progress presentation |

---

## Final Deliverables

* **Complete AstrID Temporal Dataset Pipeline and Baseline Anomaly Detection System**

* **Final Project Report (25+ pages) documenting:**
    * Astronomical temporal data processing problem domain and existing solution analysis
    * Research methodology and data engineering approach
    * Technical implementation of temporal data pipelines and anomaly detection algorithms
    * Performance evaluation results and data quality metrics
    * Limitations and future expansion opportunities
    * Critical analysis of approach and methodology
    * Rationale behind implementation choices
    * Experimentation showing strengths and weaknesses of chosen approaches

* **Oral Defense and demonstration of temporal data processing capabilities (15 minutes)**

---

## Additional Notes

* The entire project will be hosted and versioned via GitHub
* All findings must be documented in a detailed, carefully written and reviewed project report
* The final project report will include both formal description of the project, the research behind it, the practical implementation, rationale behind implementation choices, and experimentation showing strengths and weaknesses of chosen approaches
* This project serves as a foundation for potential CSCI 491 continuation in the following semester
