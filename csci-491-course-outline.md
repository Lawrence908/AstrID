# CSCI 491 Senior Research Project Outline

**Student:** Chris Lawrence
**Project Title:** Astronomical Identification: Anomaly Detection and Visualization
**Supervisor:** Prof. Luis Meneses
**Committee members:** Prof. Gregory Arkos
**Proposal:** CSCI 491 Senior Research Project (6 credits)
**Duration:** September 2, 2025 through April 30, 2026

---

## Topic Outline

The AstrID project was developed as a system for identifying stars in astronomical images using FITS data and a U-Net machine learning model, developed for a CSCI 478 course in F24 semester. This existing system can successfully localize and classify stars with reasonable accuracy, serving as a foundation for further research.

This project will extend AstrID in two key ways:

1. **Astronomical Anomaly Detection** – moving beyond static star identification to detect transient events (e.g., supernovae, brightness or size variations across multiple observations).
2. **Visualization & Public Deployment** – developing a modular web dashboard for real-time visualization of detections, enabling both scientific utility and educational/portfolio use.

This builds directly on the completed AstrID framework while introducing new machine learning research and software engineering components.

---

## Project Description

The project will employ a systematic approach combining machine learning research, software development, and astronomical data analysis to achieve the stated objectives.

**Core Research Components:**

* **Astronomical Anomaly Detection:** Investigation of time-series change detection techniques in astronomical imaging, with an emphasis on identifying supernovae and other transient anomalies. This work will extend the existing AstrID U-Net framework to support temporal analysis.

* **Public-Facing Dashboard Deployment:** Research into deploying and maintaining a production-ready web dashboard for astronomical anomaly visualization, including server architecture, and real-time data updates. This will demonstrate full-stack development capabilities and production deployment skills.

**Advanced Implementation Areas:**
* Baseline anomaly detection models extending AstrID's U-Net architecture
* Temporal analysis pipelines for multi-observation datasets
* Web-based visualization dashboard with anomaly overlays
* Automated data ingestion and result logging systems
* Confidence scoring and history tracking

**Research Methodology:** The project will employ a mixed-methods approach combining:
* Literature review of astronomical anomaly detection and related ML approaches
* Comparative analysis of existing anomaly detection solutions in astronomy
* Quantitative evaluation with precision, and astronomy-specific metrics
* Iterative development based on model performance and detection accuracy
* User experience evaluation through controlled testing of visualization features

**Evaluation Framework:** The research will be evaluated through multiple criteria including:
* Technical functionality and reliability of anomaly detection pipeline
* Performance of temporal models compared to static identification approaches
* Academic contribution to astronomical anomaly detection methodology
* Scalability and generalizability of the solution to other astronomical datasets

**Project Criteria for Different Grade Levels:**

* **A+ (Pass with honours):** Exceptional anomaly detection accuracy (>90%), fully functional web dashboard with real-time updates, comprehensive documentation, and significant academic contribution to the field.

* **B+ (Pass with distinction):** High anomaly detection accuracy (>80%), functional web dashboard with core features, thorough documentation, and clear academic contribution.

* **C (Pass):** Working anomaly detection system (>70% accuracy), basic web dashboard functionality, adequate documentation, and demonstrated learning outcomes.

* **F (Fail):** Incomplete or non-functional system, inadequate documentation, or failure to meet minimum project requirements.

**Future Research Directions:** The anomaly detection framework established in this project will enable future investigations into:
* Real-time transient event detection in astronomical surveys
* Integration with telescope observation scheduling systems
* Analytics for understanding temporal patterns in astronomical objects
* Mobile-first astronomical visualization interfaces

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
| Mid Sept | Literature review on anomaly detection in astronomy/ML; initial research log |
| Oct | Dataset preparation (temporal FITS images); design baseline anomaly detection pipeline |
| Nov | Implement baseline anomaly detection model; preliminary evaluation; begin dashboard framework |
| Dec | Prototype anomaly detection pipeline with early visualization; progress presentation |

### Spring 2026 Project Schedule:

| Date | Deliverable |
|------|-------------|
| By Jan 15 | Draft project proposal and presentation to supervisor (rehearsal) |
| By Jan 30 | Project proposal presentation and defense to committee (baseline results, pipeline, dashboard prototype) |
| Feb | Refine ML models (temporal/attention methods, hyperparameter tuning); second research log |
| March | Expand dashboard with anomaly overlays, confidence scoring, anomaly history; deploy beta version publicly |
| By April 8 | Draft final report submitted to supervisor |
| By April 15 | Revised final report submitted to supervisor |
| By April 23 | Defense version submitted to committee; rehearsal presentation to supervisor |
| By April 30 | Final project presentation and oral examination to committee and public |
| By May 30 | Final approved version accepted by all committee members |

---

## Final Deliverables

* **Complete AstrID Anomaly Detection System and Web Dashboard**

* **Final Project Report (50+ pages) documenting:**
    * Astronomical anomaly detection problem domain and existing solution analysis
    * Research methodology and machine learning approach
    * Technical implementation of temporal detection algorithms
    * Performance evaluation results and detection accuracy metrics
    * Limitations and future expansion opportunities
    * Critical analysis of approach and methodology
    * Rationale behind implementation choices
    * Experimentation showing strengths and weaknesses of chosen approaches

* **Oral Defense and demonstration of anomaly detection capabilities (20 minutes)**

---

## Additional Notes

* The entire project will be hosted and versioned via GitHub
* Deployment will be public-facing on a personal domain
* All findings must be documented in a detailed, carefully written and reviewed project report
* The final project report will include both formal description of the project, the research behind it, the practical implementation, rationale behind implementation choices, and experimentation showing strengths and weaknesses of chosen approaches