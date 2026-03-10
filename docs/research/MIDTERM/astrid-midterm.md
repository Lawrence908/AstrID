# AstrID: Supernova Detection Pipeline
## Machine Learning for Astronomical Transient Identification

**Chris Lawrence**  
**CSCI 491 - Senior Research Project**  
**Midterm Report - January 2026**
**Professor Luis Meneses**
**Professor Gregory Arkos**

---

 

1. Introduction
AstrID is a system designed to identify and analyze objects in astronomical images using machine learning and image-processing techniques. The long-term goal of the project is to build a pipeline capable of detecting changes in the night sky, such as the appearance of new objects or transient events like supernovae, by comparing astronomical images taken at different times.
Rather than focusing exclusively on theoretical research, this project emphasizes hands-on experimentation and system design. I am working through building the entire pipeline myself, including data acquisition, preprocessing, image comparison, and early machine learning components. This midterm report focuses on what has been built so far, the challenges encountered along the way, and the lessons learned during the first phase of development. The intent is not to present a finished system, but to document progress and reflect on the research process itself.

2. Background and Motivation
2.1 The Scale of Modern Astronomical Data
Modern astronomy is increasingly driven by large-scale surveys that repeatedly image wide regions of the sky. These surveys generate enormous volumes of data, far more than can be reasonably inspected by human observers. While astronomers have traditionally relied on manual inspection and targeted observation, this approach does not scale to modern datasets.
As a result, important or rare events may go unnoticed unless automated tools are used to flag them. This challenge is particularly relevant for transient phenomena, which may appear briefly and then fade. Automation is therefore not simply a convenience, but a necessity for modern astronomical analysis.
2.2 Why Machine Learning Is Relevant
Machine learning is well suited to problems involving large collections of images and subtle visual patterns. In astronomy, image-based models can learn what “normal” sky images look like and help identify deviations from that norm. These deviations may correspond to astrophysical events of interest, instrumental artifacts, or noise.
A key motivation for AstrID is the use of machine learning not as a replacement for classical astronomical techniques, but as a complementary tool. Classical methods remain essential for preparing and understanding the data, while machine learning offers a way to scale analysis and highlight candidates for further investigation.
2.3 Suitability as a Self-Study Project
This project is a strong fit for a self-directed research format because it requires work across multiple domains. It involves software engineering, data management, numerical methods, and scientific reasoning. Progress depends not only on implementing algorithms, but on understanding how different system components interact. The open-ended and iterative nature of the project aligns well with the goals of a year-long self-study.

3. Project Overview and System Design
AstrID is designed as a pipeline, meaning that data flows through a sequence of stages, with each stage performing a specific transformation or analysis. This structure makes it easier to reason about the system as a whole and to identify where problems arise.

At a high level, the pipeline begins with the acquisition of astronomical images from public archives. These images are then processed to ensure they are comparable, both spatially and numerically. Once images are aligned and normalized, they can be compared across time to identify changes. These changes are then analyzed to determine whether they correspond to real astronomical objects or artifacts.

A simplified conceptual view of the pipeline is:
Data acquisition → Image preprocessing → Image comparison → Detection → Analysis

This abstraction avoids implementation details and emphasizes the functional role of each stage. The production implementation consists of five main stages: (1) Query—retrieving observation metadata from archives, (2) Filter—ensuring same-mission compatibility, (3) Download—acquiring FITS files, (4) Organize—creating training-ready directory structures, and (5) Differencing—performing the full astronomical image differencing pipeline. The system uses YAML-based configuration for reproducibility and mission-specific customization, enabling easy extension to new data sources and scientific goals.

4. Dataset Creation and Early Experiments
4.1 The Need for a Custom Dataset
One of the first challenges encountered in this project was the lack of an existing dataset that fit the project’s goals. While many astronomical datasets are publicly available, most are not structured for pixel-level machine learning tasks that involve comparing images across time.
As a result, I constructed a custom dataset using publicly available astronomical images. This dataset pairs image data with known object information, allowing supervised learning approaches to be explored.

4.2 Supernova Catalog and Data Acquisition
The project began with a compiled catalog of 6,542 known supernovae with right ascension and declination coordinates, along with discovery dates and metadata. This catalog serves as the foundation for identifying transient events of interest. Initial data acquisition focused on downloading astronomical images from the MAST (Mikulski Archive for Space Telescopes) archive, which provides access to data from multiple space-based missions including SWIFT, GALEX, and PS1.

A pilot dataset was initially created with 19 supernovae, downloading both reference images (taken before the supernova appeared) and science images (taken after the supernova discovery). This pilot study revealed critical constraints that would shape the entire pipeline development.

4.3 Critical Discovery: Same-Mission Requirement
One of the most important discoveries from the pilot study was that image differencing requires reference and science images to come from the same telescope mission. Only 8 of 19 supernovae (42%) in the pilot dataset had usable same-mission pairs, while 11 had cross-mission data (e.g., PS1 reference with SWIFT science) that could not be used for differencing.

The root cause is that different telescopes have fundamentally different characteristics: point spread functions (PSFs), filter bandpasses, pixel scales, and noise properties. Attempting to difference images from different missions produces massive artifacts that swamp any real transient signals. This discovery led to a fundamental requirement: all differencing pairs must come from the same mission, and ideally from the same filter within that mission.

4.4 Production Dataset Development
Following the pilot study, a production pipeline was developed to scale dataset creation. The current production dataset includes 222 complete supernova pairs with 2,282 FITS files (1,170 reference images and 1,112 science images). The dataset spans multiple missions: SWIFT (17+ pairs), GALEX (105 pairs), and PS1 (additional pairs).

A key improvement was the development of a filtering stage that ensures only same-mission observations are downloaded. This filtering step reduced wasted downloads by 38% and increased the success rate from 25% to 75-100% for creating complete training pairs. The pipeline also handles compressed FITS files (.fits.gz), which was initially overlooked but proved critical for accessing the full dataset—discovering this issue led to a 296% increase in available training pairs (from 56 to 222).

4.5 FITS Files and Astronomical Image Data
Astronomical images are typically stored in the FITS (Flexible Image Transport System) format. In addition to pixel values, FITS files contain metadata describing how each pixel maps to a location in the sky. This spatial information is critical when aligning images taken at different times or by different instruments.
Working with FITS files required careful handling, as image data is not always stored in the same way across files. In some cases, image data appears in extensions rather than the primary file header, and some images contain multiple dimensions representing stacked exposures. The pipeline includes robust FITS reading that iterates through HDUs (Header Data Units) to find the first with 2D image data, and handles 3D data by taking the first slice when necessary.

4.6 Ground Truth Generation
To train and evaluate machine learning models, labeled data is required. For the initial star detection work, ground truth labels were derived from star catalogs that list known objects and their coordinates. These catalog entries were converted into pixel-based masks that indicate where stars are expected to appear in each image.

For supernova detection, ground truth is provided by the known supernova positions from the catalog. The differencing pipeline marks the known supernova pixel positions, enabling supervised learning where the model learns to distinguish real transients from artifacts and false positives.

This process highlighted the importance of accurate data representation. Small choices in how objects are labeled, such as how brightness or size is encoded, can significantly affect model performance.

4.7 Early Visualization Results
Early experiments focused on visualizing predictions alongside ground truth masks. These visual comparisons provided intuitive insight into model behavior and helped identify both successes and failure modes. For the differencing pipeline, visualization of difference images, significance maps, and detected sources alongside known supernova positions has been essential for understanding system behavior and identifying areas for improvement.

5. Machine Learning Approach
5.1 Image Segmentation as the Core Task
Rather than treating object detection as a simple classification problem, AstrID approaches it as an image segmentation task. Instead of asking whether an object is present somewhere in an image, the model predicts which pixels correspond to objects of interest. This approach provides more precise spatial information and is better suited to scientific analysis.
5.2 U-Net Architecture Intuition
The primary model architecture explored so far is a U-Net-style convolutional neural network. U-Net architectures are designed for tasks where the output is an image-like structure, such as segmentation masks. They work by capturing both large-scale context and fine-grained details, making them well suited to astronomical images where objects may be small but context matters.
5.3 Training Process and Experimentation
Training the model involved iterative experimentation with parameters such as learning rate and training duration. Rather than seeking an optimal configuration immediately, the focus was on understanding how changes in data preparation and labeling affected results.
5.4 Computational Considerations
A notable practical observation was the significant difference between CPU and GPU training performance. Training on a GPU dramatically reduced iteration time, which in turn made experimentation more feasible. This reinforced the importance of computational resources in applied machine learning research.

6. Results and Observations So Far
6.1 Star Detection Results
At this stage, the star detection model is able to identify star locations with reasonable accuracy in many cases. Predictions often align closely with known catalog data, and visual inspection shows that the model captures meaningful structure in the images. This initial work provided valuable experience with astronomical image processing and established a foundation for the more complex supernova detection task.

6.2 Supernova Detection Results
The differencing pipeline has successfully detected supernovae in the production dataset. The brightest example, SN 2014J, was detected with 2120σ significance, clearly standing out above background. Batch processing results show consistent detection across multiple supernovae, with maximum significances ranging from 412σ to 2120σ. The pipeline achieves 93.6% to 100% image overlap after WCS alignment, demonstrating robust spatial registration.

6.3 Dataset Statistics
The production dataset currently includes:
- **222 complete supernova pairs** (reference + science images)
- **2,282 FITS files** total (1,170 reference, 1,112 science)
- **Mission distribution**: SWIFT (17+ pairs), GALEX (105 pairs), PS1 (additional pairs)
- **99.1% download success rate** for supernovae with same-mission observations
- **75-100% success rate** for creating complete training pairs (improved from 25% after pipeline fixes)

6.4 Limitations and Challenges
The system exhibits clear limitations that motivate future work. Image artifacts, noise, and background variations can lead to false detections. At a 5σ detection threshold, typical difference images yield 73-629 candidate detections, but only one (the supernova) is a real transient. The remaining detections are cosmic rays, subtraction artifacts, variable stars, and host galaxy nuclei.

This false positive problem highlights the difficulty of distinguishing rare or subtle events from imperfect data. The high false positive rate demonstrates why machine learning classification is necessary as a next step.

6.5 Unexpected Detections and Scientific Potential
In some cases, the pipeline identifies objects that are not present in the reference catalog. While many of these are false positives, some may represent uncataloged variable objects or other transient events. This ambiguity underscores both the challenge and potential of automated analysis—the system may discover new phenomena while also producing many artifacts that must be filtered.

7. Challenges and Lessons Learned
7.1 Data Quality Over Model Complexity
One of the most important lessons learned is that data quality often matters more than model complexity. Improvements to labeling and preprocessing frequently produced larger gains than architectural changes. This was particularly evident in the differencing pipeline, where proper WCS alignment, PSF matching, and background subtraction were essential for meaningful results.

7.2 Infrastructure and Tooling Overhead
Another challenge was the overhead associated with tooling and infrastructure. Managing libraries, file formats, and environments required substantial effort that is often overlooked in theoretical discussions of machine learning. Working with FITS files, astronomical coordinate systems, and archive APIs introduced complexity that consumed significant development time.

7.3 Iterative Problem-Solving
Debugging machine learning systems proved fundamentally different from debugging traditional software. Model behavior emerges from interactions between data and parameters, making it necessary to adopt an experimental and iterative mindset. Similarly, debugging the differencing pipeline required understanding how each preprocessing step affected the final result, often revealing that issues in earlier stages cascaded through the pipeline.

7.4 Discovering Critical Constraints
The discovery of the same-mission requirement exemplifies how research often involves learning constraints through experimentation. The initial assumption that any two images could be differenced was incorrect, and this realization fundamentally shaped the pipeline design. Similarly, discovering that compressed FITS files (.fits.gz) were being overlooked led to a 296% increase in available training pairs (from 56 to 222).

7.5 The Importance of Modular Design
Developing a modular pipeline architecture with configuration-based execution proved invaluable. The ability to test individual stages, resume from checkpoints, and adapt to different missions without rewriting code has accelerated development and improved reproducibility. This experience reinforced the value of investing in infrastructure early, even when it seems to slow initial progress.

Overall, this phase of the project reinforced that research is rarely linear. Progress often involves revisiting earlier assumptions, discovering unexpected constraints, and refining approaches based on new insights. The iterative nature of the work has been both challenging and rewarding, providing deep understanding of both the astronomical domain and the engineering challenges of building production machine learning systems.

8. Extension Toward Supernova Detection
A major extension of the project involves detecting transient events such as supernovae. Unlike stars, which are relatively static, supernovae appear suddenly and change over time. This section describes the development of a complete image differencing pipeline for supernova detection.

8.1 Image Differencing Pipeline Overview
Image differencing is a classical astronomical technique that involves subtracting a reference image (taken before an event) from a science image (taken after the event) to highlight changes. However, simple pixel-by-pixel subtraction is inadequate—it produces massive artifacts unless images are carefully aligned, normalized, and matched in their instrumental characteristics.

The AstrID differencing pipeline consists of nine sequential stages, each addressing a specific challenge in comparing astronomical images:

1. **FITS File Loading**: Robust reading of FITS files with fallback to extension HDUs and handling of 3D data arrays
2. **WCS Alignment**: Reprojection of the science image onto the reference image's World Coordinate System (WCS) grid to achieve sub-pixel alignment
3. **Background Estimation and Subtraction**: Local background estimation using median filtering to remove varying sky levels
4. **PSF Estimation**: Measurement of point spread function full-width at half-maximum (FWHM) from bright stars in each image
5. **PSF Matching**: Convolution of the sharper image with a Gaussian kernel to match the broader PSF, eliminating bright-star residuals
6. **Flux Normalization**: Robust median-based matching of flux levels between images in the overlap region
7. **Difference Image Computation**: Subtraction of the matched reference from the normalized science image
8. **Significance Map Generation**: Creation of a signal-to-noise ratio map with uniform noise properties for consistent thresholding
9. **Source Detection**: Application of DAOStarFinder to identify candidate transients above a 5σ threshold

8.2 Critical Pipeline Components

**WCS Alignment**: Even a 1-pixel misalignment creates massive dipole residuals around every source. The pipeline uses `reproject.reproject_interp()` to transform the science image onto the reference WCS grid, achieving 93.6% to 100% overlap on processed supernovae.

**PSF Matching**: Different observations can have different seeing conditions, resulting in different PSF widths. The pipeline estimates PSF FWHM from bright stars (typically 3.5-4.0 pixels for SWIFT UVOT) and convolves the sharper image to match the broader PSF. This dramatically reduces residuals around bright stars.

**Background Subtraction**: Varying sky background levels can swamp transient signals. The pipeline uses `photutils.Background2D` with a median estimator and 50-64 pixel box sizes for local background estimation.

**Significance Maps**: Rather than thresholding raw difference images, the pipeline computes significance maps where each pixel value represents the signal-to-noise ratio. This enables consistent thresholding across images with different noise properties.

8.3 Pilot Study Results
Initial experiments using the 19-supernova pilot dataset demonstrated that meaningful transient signals can be detected when preprocessing is done correctly. The brightest supernova in decades, SN 2014J, was detected with a maximum significance of 2120σ in the U-band filter. The pipeline successfully identified the known supernova position at pixel coordinates (447.9, 555.4) among 73 candidate detections.

Batch processing of the pilot dataset showed consistent results across multiple supernovae, with maximum significances ranging from 412σ to 2120σ and detection counts from 73 to 629 candidates per image. The high number of detections reflects the false positive problem: at a 5σ threshold, many artifacts, cosmic rays, and variable stars are flagged alongside real transients.

8.4 Production Pipeline System
Following the pilot study, a production-ready pipeline system was developed with modular architecture and YAML-based configuration for reproducible dataset generation. The system consists of five main stages:

1. **Query**: Chunked MAST queries with checkpointing to handle large catalogs
2. **Filter**: Same-mission pair identification and observation filtering to ensure compatibility
3. **Download**: Smart filtering with 60-80% size reduction by avoiding incompatible observations
4. **Organize**: Creation of training-ready directory structure with reference and science subdirectories
5. **Differencing**: Full astronomical differencing pipeline execution

The configuration system uses dataclasses with validation for all pipeline parameters, enabling mission-specific configurations (SWIFT, PS1, GALEX) and reproducible dataset generation. This modular design allows easy extension to new missions and survey data.

8.5 Modular Configuration System for Training Strategy
A key feature of the production pipeline is its modular configuration system, which enables sophisticated training strategies that address the domain shift problem in astronomical machine learning. The system uses YAML-based configuration files that specify query parameters (missions, filters, temporal windows), download settings, quality thresholds, and output paths.

**Mission-Specific Training**: The configuration system allows generation of separate datasets for each mission (e.g., SWIFT-only, PS1-only, GALEX-only). This enables training mission-specific models that can learn the unique characteristics of each instrument. For example, a SWIFT UVOT model can learn the specific PSF and noise properties of that instrument, while a PS1 model can learn the multi-band characteristics of that survey.

**Filter-Specific Training**: Similarly, the system can generate datasets for specific filters within a mission (e.g., SWIFT UUU band only, or PS1 g-band only). This allows training models that specialize in particular wavelength ranges, which is important because different filters have different sensitivities and noise characteristics.

**Combined Training for Generalization**: The modular system's true power emerges when combining these specialized datasets. By training on multiple missions and filters simultaneously, a model can learn to recognize transient signals that are common across instruments while ignoring instrument-specific artifacts. This approach addresses a fundamental challenge in astronomical machine learning: models trained on one survey often fail when applied to another due to overfitting to instrument-specific characteristics.

**Reproducibility**: The YAML configuration files serve as complete specifications for dataset generation. Any dataset can be exactly reproduced by rerunning the pipeline with the same configuration file. This is critical for scientific reproducibility and for iterating on training strategies—different configurations can be tested and compared systematically.

**Example Training Strategy**: A practical training approach enabled by this system might involve:
1. Training separate models on SWIFT, PS1, and GALEX datasets to establish baseline performance
2. Training a combined model on all three datasets to learn generalizable features
3. Evaluating cross-mission performance (e.g., train on SWIFT+PS1, test on GALEX) to verify the model learns physics rather than instrument artifacts
4. Fine-tuning filter-specific models for specialized applications while maintaining the general model for broad deployment

This modular approach transforms dataset generation from a one-time effort into a flexible tool for exploring different training strategies and understanding model generalization.

8.6 Lessons from Pipeline Development
Several critical lessons emerged from developing the differencing pipeline:

- **Simple subtraction is inadequate**: Without PSF matching, background subtraction, and flux normalization, difference images are dominated by artifacts
- **Alignment must be sub-pixel accurate**: Even small offsets create dipole artifacts that mask real transients
- **Same-mission requirement is absolute**: Cross-mission differencing is fundamentally impossible due to instrumental differences
- **Filter matching matters**: Even within the same mission, comparing different filters (e.g., UVW1 vs UVM2) produces meaningless results
- **False positives dominate**: At 5σ threshold, false positives vastly outnumber true transients, necessitating machine learning classification

8.7 The False Positive Problem
A key challenge identified is that classical detection algorithms produce many false positives. At a 5σ threshold, typical difference images yield 73-629 candidate detections, but only one (the supernova) is a real transient. The remaining detections are cosmic rays, subtraction artifacts, variable stars, and host galaxy nuclei.

This false positive problem motivates the next phase: using machine learning to classify detected changes as real or spurious. A model that always predicts "bogus" would achieve ~99.99% accuracy but be scientifically useless. The correct evaluation metrics are precision, recall, F1 score, and area under the precision-recall curve (AUCPR), not accuracy.

This stage of the project demonstrates the complementary nature of classical and machine learning approaches: classical algorithms handle the well-understood physics of image alignment and differencing, while machine learning excels at the pattern recognition task of distinguishing real transients from artifacts.

9. Next Steps and Future Direction
9.1 Dataset Expansion and Training Data Generation
The current production dataset of 222 supernova pairs provides a solid foundation, but expansion is needed for robust machine learning. The immediate next step is to generate image triplets: 63×63 pixel cutouts of (science, reference, difference) images centered at detection positions. These triplets will form the input to a convolutional neural network classifier.

The training dataset will be labeled with:
- **Real=1**: Detections at known supernova positions
- **Bogus=0**: Detections at random positions, artifacts, and other false positives

Class balance is critical—approximately 50% real and 50% bogus examples will prevent the model from learning to always predict the majority class.

9.2 Machine Learning Classifier Development
The next major milestone is building a CNN architecture that takes 3-channel input (science, reference, difference triplets) and outputs a binary classification (real vs. bogus transient). The model will be trained with data augmentation (rotation, flipping) and evaluated using proper train/validation/test splits, with temporal or spatial separation to avoid data leakage.

Evaluation will focus on precision, recall, F1 score, and AUCPR rather than accuracy, given the severe class imbalance. The goal is to minimize false negatives (missing a unique supernova event is catastrophic) while maintaining reasonable precision.

9.3 Pipeline Improvements
Several pipeline enhancements are planned:
- **Pre-filtering at query stage**: Filter for same-mission pairs before downloading to reduce wasted bandwidth by ~58%
- **Filter matching prioritization**: Ensure reference and science images use matching filters within the same mission
- **Expanded temporal windows**: Consider extending reference windows to 5 years and science windows to 3 years for better coverage
- **Mission-specific optimization**: Tailor query parameters, search radii, and temporal windows per mission based on their characteristics

The modular configuration system will continue to be central to these improvements, enabling systematic exploration of different parameter combinations and training strategies. Future work will leverage this system to generate specialized datasets for ablation studies, cross-validation experiments, and domain adaptation research.

9.4 Long-term Goals
Longer-term objectives include:
- **Real-time inference API**: Deploy the trained classifier for real-time transient detection
- **Active learning loop**: Incorporate human feedback to continuously improve the model
- **Generalization testing**: Train on one survey and test on another to prove the model learns physics, not instrument artifacts
- **Extensibility**: Adapt the pipeline to different surveys, missions, and scientific goals

The modular architecture developed in the production pipeline system provides a foundation for these extensions, with YAML-based configuration enabling easy adaptation to new data sources and requirements. The ability to generate mission-specific and filter-specific datasets, then combine them for training, is particularly important for generalization testing. By systematically training on different combinations of missions and evaluating cross-mission performance, we can build models that recognize transient physics rather than instrument-specific artifacts. This approach will be essential for deploying models that work across multiple surveys and for adapting to new instruments as they come online.

10. Conclusion
This midterm report documents the progress of the AstrID project during its first phase. While the system is still under development, the work completed so far demonstrates the feasibility of using machine learning and image processing to analyze large astronomical datasets.

The project has achieved several significant milestones:
- Development of a production-ready pipeline system capable of processing 222 supernova pairs with 2,282 FITS files
- Discovery of critical constraints (same-mission requirement) that fundamentally shaped the pipeline design
- Implementation of a complete astronomical differencing pipeline with nine sequential processing stages
- Successful detection of supernovae with significances ranging from 412σ to 2120σ
- Creation of a modular, configuration-based architecture that enables reproducible dataset generation

More importantly, the project has provided valuable experience in applied research, system design, and iterative problem-solving. The lessons learned during this phase—particularly about data quality, infrastructure investment, and discovering constraints through experimentation—will directly inform the next stage of development and help shape the final outcome of the project. The foundation established in this phase, including the production pipeline system and the understanding of the false positive problem, positions the project well for the next phase: developing machine learning classifiers to distinguish real transients from artifacts.


