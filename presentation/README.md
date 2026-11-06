# Course Project Presentation: LLM Routing (TREC Track)

This directory contains the LaTeX Beamer source code for the high-level **project ideation and proposal pitch** to your professor.

## File
* [`slides.tex`](slides.tex): Self-contained LaTeX Beamer presentation (16:9 widescreen format, modern aesthetic with TikZ architectural diagram).

---

## Slide Breakdown (Abstract & High-Level)

1. **Slide 1: Title & Overview**
   * Title, sub-title, course context, and affiliation with the NIST TREC Million LLMs Track.
2. **Slide 2: Motivation & Problem Statement**
   * The proliferation of specialized models.
   * The cost and latency penalty of querying all models.
   * The Zero-Inference constraint: ranking models without calling them at test time.
3. **Slide 3: Benchmark Dataset (TREC Million LLMs Track)**
   * Overview of the 150-model benchmark.
   * Offline Discovery metadata (historical responses + token log-probabilities / confidence).
   * Development ground-truth (`qrels`).
4. **Slide 4: Abstract Methodology (Architecture Diagram)**
   * Two-stage hybrid pipeline with clean TikZ flowchart:
     * Dense Semantic Matching (embeddings)
     * Sparse Lexical Matching (BM25)
     * Query-Conditioned Uncertainty (k-NN log-probabilities)
     * Learning-to-Rank Decision Engine (XGBRanker)
5. **Slide 5: Evaluation Methodology & Rigor**
   * Evaluation metrics: nDCG@10, nDCG@20, and MRR via `ir_measures`.
   * Dense Bi-Encoder baseline comparison.
   * GroupKFold validation to guarantee zero query leakage.
6. **Slide 6: Roadmap & Expected Deliverables**
   * Project execution phases and expected takeaways.

---

## How to Compile & View

### Option A: Overleaf (Recommended - Fast & Free)
1. Go to [Overleaf](https://www.overleaf.com/).
2. Create a **New Project** $\rightarrow$ **Blank Project**.
3. Copy-paste the content of [`slides.tex`](slides.tex) into `main.tex`.
4. Click **Recompile**. Your PDF presentation is ready immediately.

### Option B: Local Compilation (Linux / Mac / Windows)
If you have TeX Live, MacTeX, or MiKTeX installed:
```bash
pdflatex presentation/slides.tex
pdflatex presentation/slides.tex
```
*(Running twice ensures Beamer resolves cross-references and total slide counts).*
