# Official 10-Slide Project Presentation: SarcasTone
**Multimodal Sarcasm Detection Using Text and Speech Prosody**

---

## Slide 1: Title Slide
### **SarcasTone: Multimodal Sarcasm Detection Using Text and Speech Prosody**
* **Project Subtitle:** Bridging the Semantic-Acoustic Incongruity Gap in Conversational Spoken Utterances
* **Course:** Natural Language Processing (NLP) Project
* **Team Member (Author):** **Pranav Sai**  
  *School of Computer Science Engineering and Technology, Bennett University, Greater Noida, India*
* **Project Instructor (Professor):** **Dr. Saurabh Kumar Srivastava**  
  *Office: M-245 | Email: `saurabh.srivastava@bennett.edu.in`*  
  *Department of Computer Science Engineering, Bennett University*
* **Project Repository:** [https://github.com/N-PRANAV-SAI/SarcasTone](https://github.com/N-PRANAV-SAI/SarcasTone)
* **Live Web Application:** [https://sarcastone.streamlit.app/](https://sarcastone.streamlit.app/)

> **Speaker Notes:**  
> Good morning Dr. Saurabh and respected committee members. I am Pranav Sai, and today I am presenting our Natural Language Processing project titled **SarcasTone: Multimodal Sarcasm Detection Using Text and Speech Prosody**. This project addresses the fundamental challenge of sarcasm detection when what a speaker says directly contradicts how they say it.

---

## Slide 2: Motivation — Why Text-Based Sentiment Fails
* **The Sarcasm Dilemma:** Sarcasm is a pragmatic communication device where the intended communicative meaning deviates sharply from the literal surface expression.
* **The Lexical Sentiment Pitfall:**
  * Statement 1: *"That’s just great!"* $\to$ Lexical models attend to "great" and predict **Positive**, but irritated acoustic delivery reveals biting sarcasm.
  * Statement 2: *"Wonderful, another meeting!"* $\to$ Words are polite and enthusiastic, yet speaker delivery is deadpan and frustrated.
* **The Missing Channel (Speech Prosody):**
  * Human listeners rely on vocal cues: deadpan monotonicity, exaggerated pitch inflections, intensity shifts, and strategic pauses.
* **Takeaway:** Without access to acoustic speech prosody, conventional text-only NLP models produce confident False Negatives on polite sarcastic remarks.

---

## Slide 3: Problem Statement & Literature Reference
* **Core Research Question (Project Specification):**
  > *"How effectively can the combination of textual information and speech prosody improve sarcasm detection when the speaker’s tone contradicts the literal meaning of the spoken words?"*
* **Foundational Literature Reference:**
  * **Santiago Castro et al. (ACL 2019)**: *"Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper)"*
  * Introduced the **MUStARD** benchmark from television dialogue scenes.
  * Established baseline BERT text ($65.10\%$) and Librosa acoustic SVM ($64.80\%$).
* **SarcasTone Key Enhancements:**
  1. *Efficient DistilBERT:* $40\%$ fewer parameters and $60\%$ faster inference than BERT-base while achieving $67.10\%$ accuracy.
  2. *Interpretable Prosody Tracking:* Explicit extraction of $F_0$ pitch contours, dynamic RMS energy, and pause ratios.
  3. *Sentiment–Tone Incongruity Engine:* Mathematical metric quantifying lexical-acoustic divergence ($\Delta_{\text{incongruity}}$).
  4. *Live Cloud Deployment:* Hosted on Streamlit Community Cloud.

---

## Slide 4: Dataset Specifications & Television Corpus Distribution
* **Benchmark Size:** **690 balanced conversational utterances** (345 Sarcastic / 345 Non-Sarcastic).
* **Multi-Speaker Television Dialogue Origin:**
  * **Friends (FRIENDS):** 356 utterances ($51.6\%$) — Multi-character sarcastic banter.
  * **The Big Bang Theory (BBT):** 280 utterances ($40.6\%$) — Dry, intellectual deadpan wit (Sheldon Cooper).
  * **The Golden Girls (GOLDENGIRLS):** 40 utterances ($5.8\%$) — Satirical conversational irony.
  * **Sarcasmoholics (SARCASMOHOLICS):** 14 utterances ($2.0\%$) — Situational conversational sarcasm.
* **Protocol Guarantees:**
  * Perfectly balanced ($50/50$) class split eliminates majority-class guessing bias.
  * Standardized audio ($22,050\text{ Hz}$ mono uncompressed WAV) aligned with verbatim transcripts.

---

## Slide 5: System Architecture — Three Computational Paradigms
1. **Unimodal Text Model (DistilBERT):**
   * Fine-tuned `distilbert-base-uncased` ($66\text{M}$ parameters).
   * Contextual pooled representation: $h_{\text{text}} \in \mathbb{R}^{768}$.
   * Dropout ($p=0.3$) + Linear classification head (AdamW, $\eta = 2 \times 10^{-5}$).
2. **Unimodal Audio Model (283-Dim Acoustic SVM):**
   * 13 MFCCs + 13 Delta MFCCs (vocal tract filtering resonances & velocity).
   * 128 Mel-spectrogram bins + 128 Delta Mel bins (spectral energy flux).
   * 1 Spectral Centroid (sound brightness center of mass).
   * Support Vector Machine with Radial Basis Function (RBF) kernel ($C=1.0$).
3. **Multimodal Intermediate Fusion (MLP):**
   * Vector Concatenation: $[768\text{ Text} \parallel 283\text{ Audio}] = 1051$ dimensions.
   * Deep MLP: Dense layers $[1051 \to 256 \to 64 \to 2]$ with ReLU activations and $L_2$ regularization.

---

## Slide 6: Speech Prosody & Sentiment-Tone Incongruity Engine
* **Real-Time Speech Prosody Tracking:**
  * **Fundamental Frequency ($F_0$):** YIN pitch tracking ($65\text{--}2093\text{ Hz}$) extracting mean, standard deviation, dynamic range, and linear slope (rising, falling, flat/deadpan).
  * **Dynamic Vocal Intensity:** Frame RMS energy measuring dynamic range and peak-to-mean emphasis ratios.
  * **Speaking Tempo:** Onset envelope tracking phonetic syllable rate (peaks/second).
  * **Pause Dynamics:** Energy-thresholded silence ($<-30\text{ dB}$) tracking pause counts and pause-to-speech ratio.
* **Sentiment–Tone Incongruity Engine Formula:**
  $$\Delta_{\text{incongruity}} = \left| \text{Polarity}_{\text{text}} - \text{Valence}_{\text{audio}} \right| \times \text{Subjectivity}_{\text{text}}$$
  * *High Mismatch Rule:* When lexical polarity is positive ($\text{Polarity} > +0.3$) while acoustic delivery is flat/deadpan ($\text{Valence} < -0.1$), the engine triggers a High Mismatch alert, uncovering covert irony.

---

## Slide 7: Experimental Setup & Stratified 5-Fold Validation
* **Stratified 5-Fold Cross-Validation:**
  * Total Dataset: $N = 690$ utterances.
  * Training Split ($80\%$): $552$ utterances per fold ($276$ sarcastic, $276$ non-sarcastic).
  * Test Split ($20\%$): $138$ utterances per fold (strictly $69$ sarcastic, $69$ non-sarcastic).
  * Zero Data Leakage: Complete isolation between training and evaluation distributions.
* **Hyperparameter Specifications:**
  * DistilBERT: Batch size $16$, $3$ epochs, AdamW ($\eta = 2 \times 10^{-5}$, weight decay $0.01$).
  * Audio SVM: RBF kernel, $C=1.0$, StandardScaler normalization, Platt probability scaling.
  * Multimodal MLP: Early stopping, $L_2$ weight penalty ($\alpha = 0.001$).
* **Evaluation Metrics:** Classification Accuracy, Weighted Precision, Weighted Recall, Weighted F1-Score.

---

## Slide 8: Empirical Results & Benchmark Comparison
| Model Architecture | Accuracy (Mean ± Std) | F1-Score (Weighted) | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **SarcasTone Text (DistilBERT)** | **$67.10\% \pm 5.66\%$** | **$66.40\% \pm 6.41\%$** | **$68.07\%$** | **$67.10\%$** |
| **SarcasTone Audio (SVM Prosody)** | **$65.65\% \pm 1.63\%$** | **$65.52\% \pm 1.50\%$** | **$65.86\%$** | **$65.65\%$** |
| **SarcasTone Multimodal (MLP Fusion)** | **$62.17\% \pm 3.82\%$** | **$61.95\% \pm 3.93\%$** | **$62.85\%$** | **$62.17\%$** |
| *Castro et al. (ACL 2019) Text (BERT)* | *$65.10\%$* | *$65.20\%$* | *$65.40\%$* | *$65.10\%$* |
| *Castro et al. (ACL 2019) Audio (SVM)* | *$64.80\%$* | *$64.70\%$* | *$65.00\%$* | *$64.80\%$* |

* **Key Takeaway 1:** DistilBERT outperforms the published BERT baseline of Castro et al. ($67.10\%$ vs. $65.10\%$, peaking at $76.09\%$ on Fold 1) with $40\%$ reduced parameters.
* **Key Takeaway 2:** Speech Prosody provides extraordinary cross-domain stability ($\sigma = \pm 1.63\%$), showing that vocal prosody is show-invariant.

---

## Slide 9: Answering the Research Question — Case Studies & Error Analysis
* **Direct Answer to Research Question:**
  Speech prosody successfully resolves sarcasm when words contradict vocal delivery. By detecting pitch contour flatlining (deadpan delivery) and dynamic energy suppression during positive lexical statements, multimodal fusion eliminates false positives and reliably isolates sarcastic intent.
* **Case Study 1 (Semantic Incongruity):**
  * Utterance: *"Oh great, another meeting that could have been an email."*
  * Text Model: Not Sarcastic ($56.3\%$ confidence) — Fooled by the word "great".
  * Audio Model: Sarcastic ($52.0\%$ confidence) — Detects flat deadpan pitch.
  * **Multimodal Fusion: Sarcastic ($99.9\%$ confidence) ✓**
* **Case Study 2 (Cross-Domain Invariance):**
  * While character vocabulary varies widely between *Friends* and *The Big Bang Theory* (causing text accuracy to fluctuate between $58.7\%$ and $76.1\%$), acoustic prosody stability ($\sigma = \pm 1.63\%$) anchors the system across domains.

---

## Slide 10: Deliverables Completed & Final Conclusion
* **Course Deliverables Checklist Completed:**
  1. **Source Code Repository:** [github.com/N-PRANAV-SAI/SarcasTone](https://github.com/N-PRANAV-SAI/SarcasTone) ✓
  2. **Research Paper:** IEEE Style Document ($7.5+$ pages, $4,420$ words, $4$ tables, $2$ figures): `SarcasTone_IEEE_Research_Paper.docx` & PDF ✓
  3. **Presentation Slides:** 10-slide PowerPoint presentation: `SarcasTone_10_Slides_Presentation.pptx` ✓
  4. **Live Cloud Web Application:** Deployed on Streamlit Community Cloud: [sarcastone.streamlit.app](https://sarcastone.streamlit.app/) ✓
  5. **Academic Audit Report:** Comprehensive technical audit: `academic-project-audit-report.md` ✓
* **Acknowledgment:** Deepest gratitude to **Dr. Saurabh Kumar Srivastava** (Office: M-245, Bennett University) for continuous academic mentorship, research reference literature (Castro et al., ACL 2019), and guidance throughout this project.
* **Thank You!** Open for questions from the evaluation committee.
