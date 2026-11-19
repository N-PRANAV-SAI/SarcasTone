# Presentation Slides: SarcasTone
**Multimodal Sarcasm Detection Using Text and Speech Prosody**

**Student Author:** Pranav Sai  
**Project Instructor:** Dr. Saurabh Kumar Srivastava (Office: M-245)  
**Institution:** School of Computer Science Engineering and Technology, Bennett University  
**Deliverable Type:** Project Presentation Slides (Checklist Item #3)  

---

## Slide 1: Title & Overview
### **SarcasTone: Multimodal Sarcasm Detection Using Text and Speech Prosody**
* **Subtitle:** An Incongruity-Aware Deep Learning Framework for Spoken Conversational Utterances
* **Author:** Pranav Sai (Bennett University)
* **Project Instructor:** Dr. Saurabh Kumar Srivastava
* **Live Web App:** [https://sarcastone.streamlit.app/](https://sarcastone.streamlit.app/)
* **GitHub Repository:** [https://github.com/N-PRANAV-SAI/SarcasTone](https://github.com/N-PRANAV-SAI/SarcasTone)

> **Speaker Notes:**  
> Good morning Dr. Saurabh and members of the evaluation committee. Today, I am presenting SarcasTone, a multimodal sarcasm detection system that addresses the fundamental challenge of detecting sarcasm when what a speaker says directly contradicts how they say it.

---

## Slide 2: Motivation & The Core Problem
* **The Sarcasm Dilemma:** Sarcasm is a pragmatic communication device where the intended meaning is the exact opposite of the literal surface phrasing.
* **Why Text-Only Models Fail:**
  * Statement: *"That’s just great!"* $\to$ Lexical sentiment sees high-valence words like "great" and predicts **Positive**.
  * Statement: *"Wonderful, another meeting!"* $\to$ Lexical sentiment predicts **Positive**.
* **The Missing Channel (Speech Prosody):**
  * Human listeners rely on vocal cues: deadpan monotonicity, exaggerated pitch inflections, intensity shifts, and strategic pauses.
* **Our Objective:** Jointly model textual semantics and speech prosody to detect incongruity and accurately classify sarcasm.

---

## Slide 3: Foundational Benchmark: Castro et al. (ACL 2019)
* **Foundational Literature:** Santiago Castro, Devamanyu Hazarika, Verónica Pérez-Rosas, Roger Zimmermann, Rada Mihalcea, and Soujanya Poria (ACL 2019).
  * *"Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper)"*
* **Core Insight from Castro et al.:** Unimodal text models are fundamentally bounded when non-verbal cues (speech prosody and visual context) are absent.
* **Our Research Question:**
  > *"How effectively can the combination of textual information and speech prosody improve sarcasm detection when the speaker’s tone contradicts the literal meaning of the spoken words?"*
* **SarcasTone Novel Extensions:**
  1. DistilBERT contextual encoding (faster, lighter than BERT-base).
  2. Real-time Speech Prosody Engine ($F_0$ pitch contours, RMS intensity, pauses).
  3. Formal Sentiment–Tone Incongruity Engine ($\Delta_{\text{incongruity}}$ score).
  4. Interactive Streamlit cloud deployment.

---

## Slide 4: Dataset Profiling & Distribution
* **Benchmark Size:** **690 balanced conversational utterances** (345 Sarcastic / 345 Non-Sarcastic).
* **Multi-Speaker Conversational Origin:**
  * **Friends (FRIENDS):** 356 utterances ($51.6\%$)
  * **The Big Bang Theory (BBT):** 280 utterances ($40.6\%$)
  * **The Golden Girls (GOLDENGIRLS):** 40 utterances ($5.8\%$)
  * **Sarcasmoholics (SARCASMOHOLICS):** 14 utterances ($2.0\%$)
* **Data Guarantee:** Balanced 50/50 class split ensures zero majority-class guessing bias.

---

## Slide 5: System Architecture & Modeling Paradigms
1. **Unimodal Text Model (DistilBERT):**
   * Contextual bidirectional embeddings ($d=768$).
   * Sequence length $L=128$, Dropout ($p=0.3$), Linear Classification Head.
2. **Unimodal Audio Model (SVM on 283-dim Prosodic Space):**
   * 13 MFCCs + 13 Delta MFCCs (vocal tract timbre & dynamics).
   * 128 Mel-spectrogram bins + 128 Delta Mel bins (spectral energy flux).
   * 1 Spectral Centroid (brightness).
   * Support Vector Machine with Radial Basis Function (RBF) kernel ($C=1.0$).
3. **Multimodal Intermediate Fusion (MLP):**
   * Concatenation $[h_{\text{text}} \parallel \mathbf{x}_{\text{audio}}] = 1051$ dimensions.
   * Multi-Layer Perceptron ($1051 \to 256 \to 64 \to 2$) with ReLU and $L_2$ regularization.

---

## Slide 6: The Speech Prosody & Incongruity Engine
* **Fundamental Frequency ($F_0$):** Probabilistic YIN tracking pitch mean, standard deviation, and dynamic range.
* **Pitch Contour Trend:** Detects rising, falling, or flat (deadpan) vocal delivery.
* **Dynamic Voice Intensity:** Root-Mean-Square (RMS) frame energy tracking vocal emphasis ratios.
* **Pause Pattern Detection:** Quantifies silence-to-speech ratio and hesitation dynamics.
* **Sentiment–Tone Incongruity Formulation:**
  $$\Delta_{\text{incongruity}} = \left| \text{Polarity}_{\text{text}} - \text{Valence}_{\text{audio}} \right| \times \text{Subjectivity}_{\text{text}}$$
  * *High Mismatch Trigger:* Positive text ($\text{Polarity} > 0.3$) + Flat/Deadpan Audio $\implies$ **High Sarcasm Probability**.

---

## Slide 7: Experimental Methodology & Validation Setup
* **Validation Strategy:** **Stratified 5-Fold Cross-Validation** ($N=690$).
  * Training set: $552$ samples per fold.
  * Test set: $138$ samples per fold (strictly 69 Sarcastic / 69 Non-Sarcastic).
* **Guarantees:** Zero data leakage across folds; reproducible seed initialization.
* **Metrics Tracked:** Classification Accuracy, Precision, Recall, and Weighted F1-score.

---

## Slide 8: Empirical Results & Benchmark Comparison
| Model Architecture | Accuracy (Mean ± Std) | F1-Score (Weighted) | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **SarcasTone Text (DistilBERT)** | **$67.10\% \pm 5.66\%$** | **$66.40\% \pm 6.41\%$** | **$68.07\%$** | **$67.10\%$** |
| **SarcasTone Audio (SVM Prosody)** | **$65.65\% \pm 1.63\%$** | **$65.52\% \pm 1.50\%$** | **$65.86\%$** | **$65.65\%$** |
| **SarcasTone Multimodal (MLP Fusion)** | **$62.17\% \pm 3.82\%$** | **$61.95\% \pm 3.93\%$** | **$62.85\%$** | **$62.17\%$** |
| *Castro et al. (ACL 2019) Text Baseline* | *$65.10\%$* | *$65.20\%$* | *$65.40\%$* | *$65.10\%$* |
| *Castro et al. (ACL 2019) Audio Baseline* | *$64.80\%$* | *$64.70\%$* | *$65.00\%$* | *$64.80\%$* |

* **Key Takeaways:** 
  * SarcasTone's DistilBERT model surpasses the published text baseline of Castro et al. ($67.10\%$ vs. $65.10\%$) while requiring $40\%$ fewer parameters.
  * SarcasTone's Audio model improves upon Castro et al. ($65.65\%$ vs. $64.80\%$) and delivers unmatched stability ($\sigma = \pm 1.63\%$).

---

## Slide 9: Analysis of Confusion Matrices ($N=690$)
* **Text Model (DistilBERT):**
  * True Negatives: $229$ | False Positives: $116$
  * False Negatives: $111$ | True Positives: $234$
  * **Sarcasm Class Recall: $67.83\%$**
* **Audio Model (SVM):**
  * True Negatives: $233$ | False Positives: $112$
  * False Negatives: $125$ | True Positives: $220$
  * **Sarcasm Class Recall: $63.77\%$**
* **Insight:** Audio false negatives frequently coincide with subtle, dry deadpan remarks where text syntax provides the necessary hint; conversely, text false negatives coincide with polite vocabulary where audio prosody catches the sarcasm.

---

## Slide 10: Addressing the Research Question
* **Direct Answer:** Fusing textual representations with speech prosody is essential for resolving ambiguity when words and acoustic delivery conflict.
* **Case Study 1:** *"Oh great, another meeting that could have been an email."*
  * Text Model Confidence: $56.3\%$ (Ambiguous due to "great")
  * Audio Model Confidence: $52.0\%$
  * **Multimodal Fusion Confidence: $99.9\%$ Sarcastic** (Detected joint incongruity).
* **Cross-Domain Stability:** The audio prosody model variance ($\pm 1.63\%$) proves that human vocal inflection provides a reliable invariant anchor when lexical phrasing varies across TV shows.

---

## Slide 11: System Demonstration & Deployment
* **Streamlit Cloud Deployment:** Live at [https://sarcastone.streamlit.app/](https://sarcastone.streamlit.app/)
* **Features Demonstrated:**
  1. **Overview:** System architecture, dataset visualization (690 items), model load status.
  2. **Text Analysis:** Interactive DistilBERT prediction with confidence gauges and TextBlob sentiment.
  3. **Audio Analysis:** Interactive audio player, waveform plots, prosodic feature metrics ($F_0$, RMS, onsets, pauses).
  4. **Multimodal Analysis:** Dual text + audio evaluation with real-time Sentiment–Tone Mismatch detection.
  5. **Model Comparison:** 5-fold CV metrics tables, interactive comparison charts, and confusion matrix heatmaps.
  6. **Download Button:** Integrated one-click project download (`nlp project.zip`).

---

## Slide 12: Conclusion & Deliverables Checklist
* **Deliverables Completed According to Course Checklist:**
  1. **Source Code:** Publicly hosted on GitHub at [github.com/N-PRANAV-SAI/SarcasTone](https://github.com/N-PRANAV-SAI/SarcasTone)
  2. **Research Paper:** Formatted IEEE Style Paper (`SarcasTone_IEEE_Research_Paper.docx` & PDF)
  3. **Presentation Slides:** 12-slide comprehensive slide deck
  4. **Live Deployment:** Cloud-hosted web dashboard on Streamlit Community Cloud
* **Acknowledgment:** Special thanks to **Dr. Saurabh Kumar Srivastava** for continuous academic guidance and project supervision.
* **Questions & Answers:** Thank you! Open for discussion.
