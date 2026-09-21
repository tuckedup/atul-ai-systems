# Atul Pandey — Systems & Machine Learning Engineering Portfolio

[![GitHub](https://img.shields.io/badge/GitHub-tuckedup-181717?style=flat&logo=github)](https://github.com/tuckedup)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C++-17-00599C?style=flat&logo=c%2B%2B&logoColor=white)](https://isocpp.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorRT](https://img.shields.io/badge/NVIDIA-TensorRT-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/tensorrt)
[![ROS2](https://img.shields.io/badge/ROS2-Jazzy-22314E?style=flat&logo=ros&logoColor=white)](https://docs.ros.org/en/jazzy/)

An engineering portfolio spanning **autonomous agent platforms**, **sim-to-real robotic perception**, **statistical verification engines**, **geospatial intelligence**, and **clinical computer vision**. Every project is built on strict engineering principles: zero synthetic metric inflation, leakage-aware validation, hardware-accelerated edge runtimes, and auditable governance.

---

## Portfolio Architecture & Domain Map

```text
                                  ┌─────────────────────────────────────────┐
                                  │      Atul Pandey — ML & Systems         │
                                  └────────────────────┬────────────────────┘
                                                       │
         ┌─────────────────────┬───────────────────────┼───────────────────────┬─────────────────────┐
         │                     │                       │                       │                     │
┌────────┴────────┐   ┌────────┴────────┐     ┌────────┴────────┐     ┌────────┴────────┐   ┌────────┴────────┐
│ Autonomous AI   │   │ Sim-to-Real &   │     │ Statistical     │     │ Geospatial &    │   │ Medical AI &    │
│ Agent Platforms │   │ Edge Perception │     │ Audit Engines   │     │ Transportation  │   │ Clinical Vision │
├─────────────────┤   ├─────────────────┤     ├─────────────────┤     ├─────────────────┤   ├─────────────────┤
│ atul-ai-systems │   │ AeroSurface-RT  │     │ ECSAE           │     │ nysdot-safety   │   │ AirwayFusion    │
│ • ForgeCode     │   │ sim2real-percep │     │ • ClinicalTrials│     │ nysdot-its-vis  │   │ OncoStruct      │
│ • RouteBench    │   │ surface-defect  │     │ • Statcheck     │     │ • H3 Hex Grid   │   │ • 3D DynUNet    │
│ • Incident Comm │   │ • TensorRT/ROS2 │     │ • GRIM/GRIMMER  │     │ • RF-DETR/ByteTr│   │ • AeroPath      │
└─────────────────┘   └─────────────────┘     └─────────────────┘     └─────────────────┘   └─────────────────┘
```

---

## 1. Autonomous AI Agent Platforms

### [`atul-ai-systems`](https://github.com/tuckedup/atul-ai-systems)
**Production-grade infrastructure for building, evaluating, and operating bounded AI agents.**

* **Core Platform (`aisys-core`)**:
  * Provider-neutral LLM gateway with structured schema validation.
  * Distributed OpenTelemetry tracing exporting directly to Arize Phoenix.
  * Durable human-in-the-loop approval gates with state serialization.
  * Cryptographically verified, hash-chained audit trails.
* **Integrated Applications**:
  1. **ForgeCode**: Autonomous coding repair loop with graph-based execution.
     * *Measured Result:* **10/10 held-out repairs passed**; 68,718 tokens; **$0.0878** total cost; **22.28s median latency**; zero runtime errors.
  2. **RouteBench**: High-throughput OpenAI-compatible model gateway.
     * Evaluates dynamic routing across quality/cost/latency thresholds.
     * Features a 3-failures-in-30s circuit breaker (60s recovery), queue timeouts, and HTTP 429 backoff.
     * *Measured Result:* Evaluated on NVIDIA RTX 4060 using vLLM FP8 KV caching and prefix caching, cutting p95 TTFT from 208.2ms to **69.06ms**.
  3. **Incident Commander**: Multi-agent site-reliability investigation system.
     * Scoped subagents with durable pause/resume across process restarts (**45/45 verified recovery runs**).
     * *Measured Result:* **15/15 exact root-cause classifications** across 5 scenarios × 3 seeds; **2.26s median time-to-root-cause**; **$0.000067 per incident**.
  4. **Operator UI**: React + Vite control surface providing live trace visualization, incident queue review, and approval dispatch.

---

## 2. Sim-to-Real Robotics & High-Throughput Edge Perception

### [`AeroSurface-RT`](https://github.com/tuckedup/AeroSurface-RT)
**Sim-to-Real Robotic Surface Perception & Edge Inference for Industrial Finishing.**

* **Problem**: Real-time semantic segmentation of aerospace/industrial panels into sandable regions, protected features, defects, and obstacles with millimetre precision.
* **Full Stack Deployment**:
  * **Model**: Custom lightweight U-Net (66k parameters) trained with domain randomization (lighting, textures, blur, noise, occlusions).
  * **ONNX & TensorRT**: Batch-1 static export numerically validated against PyTorch; TensorRT FP32 & FP16 engines built and benchmarked.
  * **C++ Runtime & ROS2**: Native C++17 ONNX Runtime engine with RAII memory management, integrated into a ROS2 Jazzy C++ node emitting masks, candidate ROIs (confidence ≥0.65, 3px erosion), and watchdog diagnostics.
* **Measured Benchmark (RTX 4060 Laptop GPU / 160×128 input)**:
  * TensorRT FP32 Latency: **0.560 ms p50** / **1.067 ms p95** (vs PyTorch CUDA 1.413 ms / ORT CPU 9.510 ms).
  * Test mIoU: **0.7561** under extreme corruptions; **0.9304** nominal validation mIoU.
  * Real-World Transfer (KolektorSDD2): Synthetic pretraining + short real fine-tuning raised defect IoU from **0.0112 → 0.2729** over scratch baselines.

### [`sim2real-perception`](https://github.com/tuckedup/sim2real-perception)
**RGB-D 7-State Surface & Tool State Segmentation with Leakage-Controlled Splits.**

* **Pipeline**: Procedural SDG (80,000 samples across 12 environments) → controlled scene-grouped splits → U-Net R34 vs. SegFormer-B0 evaluation → ONNX export → C++ ring-buffered harness.
* **Key Findings & Metrics**:
  * SegFormer-B0 achieved **0.827 test mIoU** on controlled holdout sets.
  * **Leakage Demonstration**: Highlighted that naive random splits artificially inflate test scores (0.787 naive vs 0.784 controlled) due to 11,777 leaked scene-view pairs.
  * Inference Speed: **177.1 FPS** at **7.10 ms p95 latency** via ONNX Runtime CUDA provider with a 4-slot ring buffer.

### [`surface-defect-segmentation`](https://github.com/tuckedup/surface-defect-segmentation)
**Unified Multi-Source Industrial Defect Segmentation.**

* Unifies 5 diverse industrial datasets (Severstal, NEU-DET, DAGM 2007, MVTec, SD Saliency) into a single rigorous 6-class semantic space (*crazing, inclusion, patches, pitted surface, rolled-in scale, scratches*).
* Explicitly documents annotation assumptions, excluding non-steel objects (MVTec) and coarse ellipses (DAGM) from pixel-level evaluation to guarantee evaluation integrity.

---

## 3. Statistical Auditing & Regulated Verification Engines

### [`evidence-completeness-audit-engine` (ECSAE)](https://github.com/tuckedup/evidence-completeness-audit-engine)
**Automated Statistical Inconsistency & Feasibility Audit Engine for Clinical Trials.**

* **Core Mission**: Audits controlled medical studies (ClinicalTrials.gov, AACT) for internal mathematical feasibility, reporting completeness, and transcription discrepancies.
* **Audit Rule Suites**:
  * **Partition Arithmetic**: Arm sizes, subgroup partitions, integer percentage validity, and ICEMAN interaction-test criteria.
  * **Statcheck Recomputations**: Recomputes $t$, $F$, $\chi^2$, $Q$, $r$, and $z$ test statistics from reported degrees of freedom and $p$-values.
  * **Granularity Tests**: GRIM (Granularity-Related Inconsistency of Means) and analytic GRIMMER (standard deviations).
  * **Baseline Screening**: Carlisle, Fisher, and Stouffer tests for continuous baseline variable distributions.
* **Production Architecture**:
  * FastAPI REST service with ETags, LRU/Redis caching, and Prometheus metrics.
  * Pinned GLiNER transformer extraction adapter for offline biomedical entity identification.
  * **Measured Performance**: Open-loop 20,000-request benchmark processed at 200 rps with **1.83 ms server p95** and **zero errors**.
  * **Zero Synthetic Inflation**: Exactly 220 golden regression cases; 100% rerun determinism across 4,000 paired trial comparisons.

---

## 4. Geospatial Intelligence & Intelligent Transportation Systems (ITS)

### [`nysdot-safety-risk`](https://github.com/tuckedup/nysdot-safety-risk)
**Buffalo Safety-Risk Map: Spatial-Temporal Incident Prioritization.**

* **Objective**: Transform municipal incident records, weather data, and road topologies into a continuous monthly risk surface.
* **Data & Feature Engineering**:
  * Aggregated **104,958 incident calls** (2016–2025) across **964 Uber H3 resolution-9 hexagonal cells** (48,993 cell-month records).
  * Enriched with historical meteorological data (Open-Meteo) and road network graphs (OpenStreetMap / OSMnx).
  * Strict temporal holdouts and spatial-block cross-validation to prevent geographic data leakage.
* **Modeling & Interpretability**:
  * Interpretable Negative-Binomial GLM combined with a nonlinear LightGBM Poisson model.
  * LightGBM reduced holdout MAE by **24.6%** (0.860) and RMSE by **22.9%** (1.207) over 12-month historical baselines.
  * Top 10% of predicted risk cells captured **32.3%** of all realized incidents.
  * Deployed with an interactive Streamlit GIS dashboard and SHAP global/local feature importance.

### [`nysdot-its-vision`](https://github.com/tuckedup/nysdot-its-vision)
**Edge Queue Analysis & Stopped-Vehicle Incident Replay.**

* **Architecture**: Real-time traffic camera pipeline integrating RF-DETR Nano, ByteTrack tracker, and Supervision.
* **Features**:
  * Interactive 4-point planar homography calibration mapping road pixels to real-world metric space.
  * Incident finite state machines (FSM) detecting stopped vehicles with queue-suppression guards.
  * Motion-guard watchdogs that invalidate event detection if camera sway or vibration occurs.
  * Local FastAPI service for clip ingestion and Streamlit event replay viewer.

---

## 5. Medical AI & Clinical Engineering

### [`AirwayFusion`](https://github.com/tuckedup/AirwayFusion)
**3D Anatomical Airway Segmentation, Centerline Analysis & Topology Repair.**

* 3D volumetric airway tree segmentation using DynUNet architectures trained on public AeroPath CT volumes.
* Integrates centerline extraction, branch detection, stenosis measurement, and geometric offline QC.
* Transparent reporting: Documents real AeroPath test metrics (Dice 0.3311, HD95 140.48 mm, tree-length detection 0.4703) without synthetic inflation, highlighting the genuine complexity of sparse clinical airway segmentation.

### [`OncoStruct`](https://github.com/tuckedup/OncoStruct)
**Clinical Oncology Report Structuring & Evidence Extraction.**

* Strict clinical data safety protocol: Zero external API calls, zero real patient text in logs/commits, and mandatory character-level evidence spans for every extracted clinical attribute.

---

## 6. Data & Predictive Analytics

### [`DIC_Project`](https://github.com/tuckedup/DIC_Project)
**Macroeconomic & Higher Education Job Market Trend Simulator.**

* Models employment volatility and market demand by combining scraped university enrollment statistics, industry indicators, and U.S. Bureau of Labor Statistics (BLS) APIs.

---

## Technical Skills Matrix

| Domain | Technologies & Frameworks |
|:---|:---|
| **Autonomous Agents & LLMs** | OpenTelemetry, Arize Phoenix, LangGraph, vLLM, FP8 KV Caching, Prefix Caching, Typed Tools, Audit Hash-Chaining |
| **Edge Perception & Robotics** | PyTorch, TensorRT (FP32/FP16), ONNX Runtime, ROS2 Jazzy, OpenCV, RF-DETR, ByteTrack, Supervision |
| **Languages** | Python (3.10–3.13), C++ (C++17), SQL, Bash / PowerShell |
| **Backend & Deployment** | FastAPI, Uvicorn, Docker, Docker Compose, Redis, PostgreSQL, Streamlit, React / Vite, Prometheus, Grafana |
| **Geospatial & Analytics** | Uber H3, GeoPandas, OSMnx, Shapely, LightGBM, Scikit-learn, SciPy, SHAP, Statsmodels |
| **Engineering Rigor** | Leakage-aware validation, deterministic rerun harnesses, property testing, CI/CD, synthetic-to-real domain transfer |

---

## Repository Index

| Repository | Focus Area | Key Technologies |
|:---|:---|:---|
| [**`atul-ai-systems`**](https://github.com/tuckedup/atul-ai-systems) | Autonomous Agents & Gateway | Python, OpenTelemetry, vLLM, React, FastAPI |
| [**`AeroSurface-RT`**](https://github.com/tuckedup/AeroSurface-RT) | Robotic Surface Perception | PyTorch, TensorRT, C++17, ROS2 Jazzy |
| [**`evidence-completeness-audit-engine`**](https://github.com/tuckedup/evidence-completeness-audit-engine) | Statistical Trial Auditing | Python, GLiNER, FastAPI, Docker, Redis |
| [**`nysdot-safety-risk`**](https://github.com/tuckedup/nysdot-safety-risk) | Geospatial Incident Risk | Uber H3, LightGBM, Streamlit, GeoJSON |
| [**`nysdot-its-vision`**](https://github.com/tuckedup/nysdot-its-vision) | Edge Traffic Video Analytics | RF-DETR, ByteTrack, Homography, Streamlit |
| [**`sim2real-perception`**](https://github.com/tuckedup/sim2real-perception) | RGB-D Sim-to-Real Vision | PyTorch, SegFormer, ONNX Runtime, CUDA |
| [**`surface-defect-segmentation`**](https://github.com/tuckedup/surface-defect-segmentation) | Multi-Dataset Defect Vision | PyTorch, TensorRT, Severstal, NEU-DET |
| [**`AirwayFusion`**](https://github.com/tuckedup/AirwayFusion) | 3D Medical Airway Vision | DynUNet, AeroPath, 3D Topology, MONAI |
| [**`DIC_Project`**](https://github.com/tuckedup/DIC_Project) | Labor Market Trend Modeling | Python, BLS APIs, Web Scraping |

---

*Authored by [Atul Pandey](https://github.com/tuckedup). Reach out on [LinkedIn](https://linkedin.com) or via [GitHub](https://github.com/tuckedup).*
