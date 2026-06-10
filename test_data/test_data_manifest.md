# Educational Diagram Decomposition and PPT Generation Platform — Test Dataset Manifest

This test dataset contains **27 high-resolution sample diagrams** generated programmatically to validate the detection, segmentation, OCR, and PowerPoint reconstruction capabilities of the platform.

## 🛠️ Dataset Generation & Attribution
* **Method:** Generated synthetically using a custom Pillow/OpenCV drawing engine in [generate_test_dataset.py](file:///C:/Work/PPT%20Generation%20application/generate_test_dataset.py).
* **Source/Attribution:** Built by the AI Engineering team. Every diagram uses a premium HSL-curated color palette (soft pastels, off-white background, clean text, and slate gray arrows) to emulate modern textbook illustrations.
* **Layouts:** Features precise bounding boxes, text frames, geometric shapes, and connector arrows designed specifically to test local CV edge-finding, OCR bounding-box associations, and background erasure algorithms.

---

## 📂 Category Breakdown

### 🧬 1. Biology Diagrams
Designed to validate irregular outline segmentation (cell walls/membranes) and label-pointer distance-clustering.

| Filename | Description | Key Features to Test |
| :--- | :--- | :--- |
| [`biology_digestive_system.png`](file:///C:/Work/PPT%20Generation%20application/test_data/biology_digestive_system.png) | Mouth, Esophagus, Stomach, Liver, Small/Large Intestine layout. | Left/right side label coordinate mappings with arrows. |
| [`biology_human_heart.png`](file:///C:/Work/PPT%20Generation%20application/test_data/biology_human_heart.png) | Left/Right Atrium/Ventricles, Aorta, Vena Cava. | Deep overlap of structures and nested boundaries. |
| [`biology_plant_cell.png`](file:///C:/Work/PPT%20Generation%20application/test_data/biology_plant_cell.png) | Green cell wall (octagon), vacuole, nucleus, and chloroplasts. | Sharp green contours, nested cell wall vs. cell membrane. |
| [`biology_animal_cell.png`](file:///C:/Work/PPT%20Generation%20application/test_data/biology_animal_cell.png) | Soft circular cell membrane with organelles. | Irregular soft oval boundaries and inner details. |

### ⚡ 2. Physics & Earth Science Diagrams
Designed to validate linear rays, mathematical labels, orbital curves, and vector elements.

| Filename | Description | Key Features to Test |
| :--- | :--- | :--- |
| [`physics_solar_system.png`](file:///C:/Work/PPT%20Generation%20application/test_data/physics_solar_system.png) | Sun with planets along concentric orbital arcs. | Concentric orbital curves (should not be detected as image objects). |
| [`physics_water_cycle.png`](file:///C:/Work/PPT%20Generation%20application/test_data/physics_water_cycle.png) | Ocean, mountains, clouds, sun, rain, and cyclic process labels. | Complex landscape layout, overlapping rain lines. |
| [`physics_electric_circuit.png`](file:///C:/Work/PPT%20Generation%20application/test_data/physics_electric_circuit.png) | Wire loops with battery, bulb, open switch, and current flow. | Thin wire detection (should be ignored), symbol crops. |
| [`physics_reflection_of_light.png`](file:///C:/Work/PPT%20Generation%20application/test_data/physics_reflection_of_light.png) | Normal, incident/reflected ray angles relative to mirror. | Dotted normal line, thin angle arcs, Greek/letter labels (`i`, `r`). |
| [`physics_pendulum_system.png`](file:///C:/Work/PPT%20Generation%20application/test_data/physics_pendulum_system.png) | Damped pendulum bob showing gravity and tension vectors. | Diagonal string path, vector force labels. |

### 📊 3. Flowcharts
Designed to validate standard flowchart symbols (ovals, rectangles, diamonds) and structured connection grids.

| Filename | Description | Key Features to Test |
| :--- | :--- | :--- |
| [`flowchart_student_admission_process.png`](file:///C:/Work/PPT%20Generation%20application/test_data/flowchart_student_admission_process.png) | Start -> Form -> Eligibility (Decision) -> Pay/Reject -> End. | Decision paths (Yes/No branches) and merged nodes. |
| [`flowchart_software_development_lifecycle.png`](file:///C:/Work/PPT%20Generation%20application/test_data/flowchart_software_development_lifecycle.png) | Sequential SDLC blocks with feedback loop. | Long return loops spanning multiple horizontal nodes. |
| [`flowchart_order_processing.png`](file:///C:/Work/PPT%20Generation%20application/test_data/flowchart_order_processing.png) | Check stock flowchart with Yes/No routes. | Standard vertical flow node alignment. |
| [`flowchart_temperature_control_loop.png`](file:///C:/Work/PPT%20Generation%20application/test_data/flowchart_temperature_control_loop.png) | Read Temp -> Decision -> Action -> Delay loop. | Closed feedback loops. |

### 🎨 4. Infographics
Designed to validate tabular layout structures, title headers, color-coded sections, and multi-line descriptions.

| Filename | Description | Key Features to Test |
| :--- | :--- | :--- |
| [`infographic_parts_of_speech.png`](file:///C:/Work/PPT%20Generation%20application/test_data/infographic_parts_of_speech.png) | Vertical card columns representing Noun, Verb, Adjective, Adverb. | Multi-line text boxes with card backgrounds. |
| [`infographic_learning_process.png`](file:///C:/Work/PPT%20Generation%20application/test_data/infographic_learning_process.png) | 4-step learning timeline (Read, Process, Practice, Teach). | Header banners aligned with text block descriptions. |
| [`infographic_photosynthesis.png`](file:///C:/Work/PPT%20Generation%20application/test_data/infographic_photosynthesis.png) | Sunlight, H2O, CO2, O2, Glucose flows around plant pot. | Plant shape segmentation, chemistry symbol layouts. |
| [`infographic_water_conservation.png`](file:///C:/Work/PPT%20Generation%20application/test_data/infographic_water_conservation.png) | Droplet-centered layout with circular outer strategy nodes. | Concentric radial layout and pointing lines. |
| [`infographic_chemistry_alkali_metals.png`](file:///C:/Work/PPT%20Generation%20application/test_data/infographic_chemistry_alkali_metals.png) | Grid elements representing Group 1 elements. | Regular grid segmentation, bold symbol detection. |
| [`infographic_study_methods.png`](file:///C:/Work/PPT%20Generation%20application/test_data/infographic_study_methods.png) | 2x2 grid representing scientific studying techniques. | Large box layouts with nested title and descriptive text. |
| [`infographic_nitrogen_cycle.png`](file:///C:/Work/PPT%20Generation%20application/test_data/infographic_nitrogen_cycle.png) | Soil/bacteria nitrogen cycle loop. | Circular loop of label nodes. |

### 📝 5. Difficult Cases (Scans & Mixed Content)
Designed to challenge the classifier, detector, and OCR components with overlapping text, double-column layouts, formulas, and large text paragraphs.

| Filename | Description | Key Features to Test |
| :--- | :--- | :--- |
| [`difficult_textbook_scan_1.png`](file:///C:/Work/PPT%20Generation%20application/test_data/difficult_textbook_scan_1.png) | Textbook page simulation: title, body text, cell membrane diagram, footnote. | Distinguishing between body text paragraph blocks and label blocks. |
| [`difficult_textbook_scan_2.png`](file:///C:/Work/PPT%20Generation%20application/test_data/difficult_textbook_scan_2.png) | Double-column text: math equations (Snell's/Cauchy's) and light dispersing prism. | Multi-column layout separation, inline equations, rainbow colors. |
| [`difficult_textbook_scan_3.png`](file:///C:/Work/PPT%20Generation%20application/test_data/difficult_textbook_scan_3.png) | Faraday's Induction: Formula box, bar magnet, wire coil, galvanometer. | Deflected meter needle, wire connections, bold equations. |
| [`difficult_textbook_scan_4.png`](file:///C:/Work/PPT%20Generation%20application/test_data/difficult_textbook_scan_4.png) | Chemical Equilibrium: Beaker diagram, catalyst pouring, math equation, warnings. | Multi-element physics scan with large safety block warnings. |
| [`difficult_mixed_content_1.png`](file:///C:/Work/PPT%20Generation%20application/test_data/difficult_mixed_content_1.png) | Quiz page: Question header, circuit diagram, multiple choice options grid, instructions. | Mixing standard questions, diagram sub-elements, and option boxes. |
| [`difficult_mixed_content_2.png`](file:///C:/Work/PPT%20Generation%20application/test_data/difficult_mixed_content_2.png) | History Timeline: Central axis line, 5 historical cards (ENIAC to Google), body paragraph. | Long horizontal timeline, tight spacing of date labels, overlapping lines. |
| [`difficult_mixed_content_3.png`](file:///C:/Work/PPT%20Generation%20application/test_data/difficult_mixed_content_3.png) | Renaissance Page: Historical intro paragraph, event timeline, details paragraph. | Large mixed text page with timelines embedded in paragraphs. |

---

## 🎯 Validation Verification Plan
Use these sample images to test:
1. **Contour Detection:** Verify that text labels are cleanly separated from shape regions.
2. **Text Erasure:** Ensure that after generating the PPT slide, the text labels on the slide are fully editable text boxes and that the backing template `background.png` has the original text cleanly erased (replaced by background color) to prevent double-rendering.
3. **Pluggable OCR:** Verify OCR reading accuracy when using EasyOCR/Tesseract vs. local heuristics fallback.
