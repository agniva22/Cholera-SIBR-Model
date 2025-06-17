# Cholera-SIBR-Model

readme_content = """# Cholera SIBR Model Parameter Estimation

This repository contains Python scripts to calibrate a mechanistic SIBR (Susceptible–Infected–Bacteria–Recovered) model to weekly cholera case data using

1. **Deterministic maximum-likelihood estimation** (via `scipy.optimize.minimize`), and  
2. **Bayesian uncertainty quantification** (via Delayed-Rejection Adaptive Metropolis MCMC using `pymcmcstat`).  

---

## Table of Contents

- [Background](#background)  
- [Repository Structure](#repository-structure)  
- [Dependencies](#dependencies)  
- [Installation](#installation)  
- [Data](#data)  
- [Usage](#usage)  
  - [1. Deterministic Calibration (MLE)](#1-deterministic-calibration-mle)  
  - [2. Bayesian Calibration (DRAM MCMC)](#2-bayesian-calibration-dram-mcmc)  
- [Results & Plots](#results--plots)  
- [Configuration](#configuration)  
- [License](#license)  

---

## Background

The SIBR model captures cholera transmission dynamics via:

- **S**: susceptible human population  
- **I**: infected population  
- **B**: environmental bacterial concentration  
- **R**: recovered population  

Six key parameters are estimated:

1. Environmental transmission rate, σₑ  
2. Human-to-human transmission rate, σₕ  
3. Water-sanitation efficacy, ω  
4. Environmental decay/disinfection rate, δ  
5. Maximum recovery rate, ϕ₁  
6. Hospital-bed ratio, a  

---

## Repository Structure

├── data
├── scripts
├── requirements.txt
|── README.md 
---

## Dependencies

- Python 3.8+  
- NumPy  
- SciPy  
- pandas  
- matplotlib  
- pymcmcstat  

---

## Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/yourusername/cholera-sibr-calibration.git
   cd cholera-sibr-calibration

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
