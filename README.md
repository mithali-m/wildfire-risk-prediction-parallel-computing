# High-Performance Wildfire Risk Prediction: A Parallel Computing Approach
## Overview

Wildfire ignition prediction is an important challenge in environmental safety and disaster prevention. Increasing wildfire frequency due to climate change has made predictive analytics a critical tool for early warning systems and resource planning.

This project demonstrates how High-Performance Computing (HPC) techniques can accelerate wildfire prediction pipelines by parallelizing data processing, feature engineering, and model training on a large-scale meteorological dataset.

Using the FireCastRL US Wildfire Dataset, which contains 9.5 million daily observations across 37,000+ locations in the United States, we implement and benchmark multiple parallel computing frameworks including Dask, Joblib, and built-in multithreading in machine learning libraries. 

The goal is not only to build predictive models, but also to analyze computational speedups, efficiency, and scaling behavior under different CPU configurations.

## Key Objectives

This project focuses on three major objectives:

1. Computational Optimization - Parallelize temporal feature engineering and model training using multiple cores to reduce computation time.
2. HPC Theory Validation - Validate Amdahl’s Law by measuring theoretical vs observed speedup in parallel workloads.
3. Parallel Machine Learning - Benchmark different parallel training strategies for tree-based models including:
- XGBoost
- Random Forest

These models are evaluated under several parallel computing frameworks.

## Dataset
### FireCastRL US Wildfire Dataset

- Source: Kaggle FireCastRL Lab
- Size: 9.5 million records
- Time span: 2013 – 2025
- Spatial coverage: 37,098 unique locations across the continental US

Each record contains:
- meteorological variables
- fuel moisture indices
- atmospheric indicators
- wildfire ignition label

Examples of features include:
- Temperature (tmmn, tmmx)
- Humidity (rmin, rmax)
- Precipitation
- Wind speed
- Vapor pressure deficit
- Energy Release Component
- Burning Index
- Fuel moisture indices

The target variable is Wildfire ignition (binary classification).

## System Architecture

The project pipeline follows a typical HPC-enabled machine learning workflow.

```declaration
Raw Wildfire Dataset
        │
        ▼
Parallel Data Loading (Dask)
        │
        ▼
Feature Engineering
60-day rolling statistics + lag features
(Joblib Parallelization)
        │
        ▼
Model Training
XGBoost / Random Forest
        │
        ▼
Parallel Hyperparameter Search
(Joblib CV / Dask)
        │
        ▼
Performance Benchmarking
Speedup | Efficiency | Overhead
```
