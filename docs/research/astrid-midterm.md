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
This abstraction avoids implementation details and emphasizes the functional role of each stage. A visual diagram of this pipeline would be appropriate to include here.

4. Dataset Creation and Early Experiments
4.1 The Need for a Custom Dataset
One of the first challenges encountered in this project was the lack of an existing dataset that fit the project’s goals. While many astronomical datasets are publicly available, most are not structured for pixel-level machine learning tasks that involve comparing images across time.
As a result, I constructed a custom dataset using publicly available astronomical images. This dataset pairs image data with known object information, allowing supervised learning approaches to be explored.
4.2 FITS Files and Astronomical Image Data
Astronomical images are typically stored in the FITS (Flexible Image Transport System) format. In addition to pixel values, FITS files contain metadata describing how each pixel maps to a location in the sky. This spatial information is critical when aligning images taken at different times or by different instruments.
Working with FITS files required careful handling, as image data is not always stored in the same way across files. In some cases, image data appears in extensions rather than the primary file header, and some images contain multiple dimensions representing stacked exposures.
4.3 Ground Truth Generation
To train and evaluate machine learning models, labeled data is required. In this project, ground truth labels were derived from star catalogs that list known objects and their coordinates. These catalog entries were converted into pixel-based masks that indicate where stars are expected to appear in each image.
This process highlighted the importance of accurate data representation. Small choices in how objects are labeled, such as how brightness or size is encoded, can significantly affect model performance.
4.4 Early Visualization Results
Early experiments focused on visualizing predictions alongside ground truth masks. These visual comparisons provided intuitive insight into model behavior and helped identify both successes and failure modes. This stage is well suited for including example figures that overlay predicted and expected star locations.

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
6.1 Successful Outcomes
At this stage, the model is able to identify star locations with reasonable accuracy in many cases. Predictions often align closely with known catalog data, and visual inspection shows that the model captures meaningful structure in the images.
6.2 Limitations and Errors
The model also exhibits clear limitations. Image artifacts, noise, and background variations can lead to false detections. These issues highlight the difficulty of distinguishing rare or subtle events from imperfect data.
6.3 Unexpected Detections
In some cases, the model identifies objects that are not present in the reference catalog. While these may be false positives, they also raise the possibility of detecting uncataloged or variable objects. This ambiguity underscores both the challenge and potential of automated analysis.

7. Challenges and Lessons Learned
One of the most important lessons learned so far is that data quality often matters more than model complexity. Improvements to labeling and preprocessing frequently produced larger gains than architectural changes.
Another challenge was the overhead associated with tooling and infrastructure. Managing libraries, file formats, and environments required substantial effort that is often overlooked in theoretical discussions of machine learning.
Debugging machine learning systems also proved fundamentally different from debugging traditional software. Model behavior emerges from interactions between data and parameters, making it necessary to adopt an experimental and iterative mindset.
Overall, this phase of the project reinforced that research is rarely linear. Progress often involves revisiting earlier assumptions and refining approaches based on new insights.

8. Extension Toward Supernova Detection
A major extension of the project involves detecting transient events such as supernovae. Unlike stars, which are relatively static, supernovae appear suddenly and change over time.
Early work in this area focused on image differencing, which involves subtracting a reference image from a later image to highlight changes. This process requires careful alignment and normalization to avoid introducing artifacts.
Initial experiments using a small pilot dataset demonstrated that meaningful transient signals can be detected when preprocessing is done correctly. These experiments also revealed strict requirements, such as the need for reference and science images to come from the same instrument and filter.
This stage of the project is a natural bridge between star detection and more general anomaly detection.

9. Next Steps and Future Direction
In the next phase of the project, the dataset will be expanded to include a larger number of transient events. Improving the robustness of image differencing and reducing false positives are immediate priorities.
A longer-term goal is to introduce machine learning models that classify detected changes as real or spurious. This will require carefully constructed training data and appropriate evaluation metrics.
Ultimately, the aim is to develop a flexible and extensible pipeline that can adapt to different datasets and scientific goals.

10. Conclusion
This midterm report documents the progress of the AstrID project during its first phase. While the system is still under development, the work completed so far demonstrates the feasibility of using machine learning and image processing to analyze large astronomical datasets.
More importantly, the project has provided valuable experience in applied research, system design, and iterative problem-solving. The lessons learned during this phase will directly inform the next stage of development and help shape the final outcome of the project.

