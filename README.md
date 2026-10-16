# 🔍 Fake News Detector AI

A production-ready, multi-layered misinformation detection system combining a high-accuracy Machine Learning ensemble with a **Retrieval-Augmented Generation (RAG)** pipeline for real-time, evidence-based news verification.

---

## 📖 Overview

The **Fake News Detector AI** is a state-of-the-art misinformation detection system that tackles fake news from two complementary angles:

1. **Linguistic Analysis (ML):** A 5-model Voting Ensemble trained on 100,000+ articles detects the *writing style* of fake news — sensationalism, conspiracy language, clickbait patterns, and structural anomalies.
2. **Factual Verification (RAG):** A Retrieval-Augmented Generation pipeline queries live news via GNews, injects the results into a Groq LLaMA 3 prompt, and cross-references the article's claims against real-time evidence.

This dual approach means the system can catch both *stylistically* fake articles (clickbait, conspiracy blogs) and *factually* fake articles (fabricated events, outdated claims presented as new).

---

## ✨ Key Features

### 🧠 RAG-Powered Real-Time Verification
- Extracts keywords from user input and queries the **GNews API** for live headlines.
- Injects live news context directly into the **Groq LLaMA 3** prompt.
- The AI cross-references article claims against breaking news evidence.
- **Intelligent Fallback:** Automatically switches to standalone style-analysis mode if no live news is found.

### 🌐 Multi-Source Intelligence & Caching (v7.0 Upgrades)
- **Google Fact Check API:** Directly queries Google's ClaimReview database to find if claims have already been verified by trusted orgs (Reuters, AFP, Alt News, etc.).
- **Google Safe Browsing API:** Extracts URLs from article text and scans them for malware, phishing, and social engineering threats.
- **Source Credibility Database:** Built-in domain reputation system covering 80+ Indian and international news sources, graded by credibility tier, category, and bias.
- **Claim Caching Engine:** In-memory TTL cache with normalized hashing that sits in front of external APIs, providing a **160x speedup** on repeated viral claims.

### 🎓 Advanced Features (v8.0 Upgrades)
- **Educator Mode:** Toggle-based step-by-step pipeline transparency showing exactly how each analysis step reached its conclusion — text preprocessing, ML ensemble voting, meta-feature extraction, AI cross-reference, live news retrieval, and fact-check results.
- **Voice Input (Groq Whisper):** Upload WhatsApp voice notes, audio files, or record directly from the browser microphone. Audio is transcribed via Groq Whisper (supports Hindi, English, 50+ languages) and fed into the full analysis pipeline.
- **India Threat Scanner:** 9-category threat detection engine with 150+ India-specific keywords covering UPI/banking fraud, fake government schemes, WhatsApp forward patterns, health misinformation, communal triggers, fake job scams, religious manipulation, India-specific conspiracies, and fake reward/lottery schemes.
- **Community Dashboard:** Live platform-wide stats bar showing total analyses, fake detection rate, average confidence, and today's activity — powered by anonymized aggregated data.

### 🤖 5-Model Voting Ensemble
- **Soft Voting Classifier** combining Logistic Regression, Random Forest, SGD Classifier, LinearSVC (calibrated), and LightGBM.
- Averaged probability outputs with an **F1-optimized decision threshold** (0.552).
- Custom class weights `{FAKE: 1.0, REAL: 1.5}` to minimize false positives on real articles.

### 🧬 Hybrid Feature Engineering
- **TF-IDF Vectorization:** 25,000 n-gram features (unigrams to trigrams) with sublinear TF scaling.
- **20 Custom Meta-Features:** Extracted from raw text before cleaning — captures signals like `all_caps_ratio`, `conspiracy_score`, `sensationalism_score`, `flesch_kincaid_grade`, `clickbait_score`, and `has_attribution`.
- **NUM Token Masking:** Numbers are normalized to `<NUM>` tokens during preprocessing to preserve sentence structure for TF-IDF.

### 🛡️ Out-of-Distribution (OOD) Detection
- Computes TF-IDF centroid distance during inference.
- Flags non-news inputs (recipes, code, poetry) before the model makes a blind guess.
- Dynamic weight shifting reduces ML influence on OOD articles.

### 🏥 Production Monitoring
- SQLite-based prediction tracking (`analysis_history.db`).
- Automated health alerts for drift detection, OOD spikes, and confidence anomalies.
- Comprehensive evaluation reports (JSON/PDF).

### 🎨 Modern Web Frontend
- Responsive dashboard with real-time analysis progress indicators.
- Dynamic "AI + LIVE NEWS" badge when RAG mode activates.
- Adjustable sensitivity slider with live numeric tooltip.
- User authentication and analysis history.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   User Input (Article Text)          │
└──────────────────────────┬──────────────────────────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐
  │  ML Pipeline │  │ RAG Pipeline│  │ Model Monitor    │
  │              │  │ (Smart      │  │ (Health Tracking)│
  │ TF-IDF +    │  │  Verify)    │  │                  │
  │ 20 Meta     │  │             │  │ • Drift Detection│
  │ Features    │  │ Step 1:     │  │ • OOD Frequency  │
  │      │      │  │  Keywords   │  │ • Confidence     │
  │      ▼      │  │      │      │  │   Distribution   │
  │ 5-Model     │  │      ▼      │  └──────────────────┘
  │ Voting      │  │ Step 2:     │
  │ Ensemble    │  │  GNews API  │
  │      │      │  │      │      │
  │      ▼      │  │      ▼      │
  │ REAL/FAKE   │  │ Step 3:     │
  │ + Confidence│  │  Inject     │
  └──────────────┘  │  Context   │
                    │      │      │
                    │      ▼      │
                    │ Step 4:     │
                    │  Groq LLaMA │
                    │      │      │
                    │      ▼      │
                    │ VERDICT +   │
                    │ Evidence    │
                    └─────────────┘
