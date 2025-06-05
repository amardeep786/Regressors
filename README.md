# 🔬 Performance Analysis & Regression Repository

This repository contains comprehensive tools for analyzing application performance, resource utilization, and processing time prediction across multiple domains.

---

## 📂 Repository Structure

### 🔄 **AllCombination**
*System Resource Monitoring & Multi-Application Analysis*

Contains code for generating and testing **all possible combinations** of 3 applications with varying instance counts. Captures comprehensive system metrics including:

- **CPU Usage** - Per-core utilization during multi-app execution
- **GPU Usage** - Graphics processing load analysis  
- **Application-wise RAM** - Memory consumption per application instance
- **System Performance** - Cross-application resource interference patterns

**Key Features:**
- Automated combination generation for workload testing
- Real-time system metric collection
- Resource contention analysis across applications

---

### 🤖 **Individual_Regressor**
*Per-Instance Processing Time Prediction*

Machine learning regressors for predicting processing times across three domains:

| Regressor Type | Purpose | Binning Strategy |
|----------------|---------|------------------|
| **detectRegressor** | Object detection time prediction | 5ms bins |
| **predictRegressor** | Instance segmentation time prediction | 5ms bins |
| **speechRegressor** | Speech-to-text conversion analysis | N/A |

**Performance Metrics:** MSE, R² scores with granular bin-level analysis

---

### ⏱️ **Waiting_Time**
*Process Queue Analysis*

Contains code and datasets for analyzing **application waiting times** in the system ready queue:

- **Proc System Calls** - Direct kernel-level timing measurements
- **Queue Analysis** - Ready queue delay patterns by application type
- **Temporal Analysis** - Waiting time distributions and trends

**Data Sources:** Linux `/proc` filesystem integration for real-time process monitoring

---

### 🎤 **speech Regressor**
*Speech Processing Performance Analysis*

Specialized regression models for speech-to-text conversion timing and accuracy metrics.

---

## 🛠️ Usage Overview

