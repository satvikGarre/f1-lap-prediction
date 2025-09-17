***

# F1 Lap Time Prediction Using Machine Learning and Deep Learning

## Overview

This project leverages **FastF1**, a Python package for accessing Formula 1 telemetry data, combined with machine learning (ML) and deep learning (DL) techniques to predict lap times of F1 drivers. Using real race data collected from multiple Grand Prix events, the project demonstrates:

- Data extraction and preprocessing of F1 race telemetry
- Feature engineering, including lap sector times and tyre compounds
- Sequence modeling with LSTM neural networks to predict next lap times based on previous laps
- Comparison with a classic machine learning baseline (Random Forest)
- Quantitative performance evaluation with RMSE, MAE, and R² metrics
- Visualization of actual vs predicted lap times

***

## Project Motivation

Lap time prediction is critical in Formula 1 for race strategy optimization, tyre management, and performance analysis. This project builds a data-driven pipeline that provides insights into the factors that influence a driver’s performance during a race and explores advanced sequential modeling techniques to forecast lap times accurately.

***

## Dataset

- Data is collected using the [FastF1](https://github.com/theOehrly/Fast-F1) Python package that retrieves official F1 telemetry, lap times, and session data.
- Multiple race sessions are combined (e.g., 2023 & 2024 Monza and Silverstone Grand Prix races) to create a robust training dataset for the selected driver Max Verstappen (code: VER).
- Key features include:
  - LapTime, Sector1Time, Sector2Time, Sector3Time (all converted to seconds)
  - Tyre compound codes
- Data cleaning involves removal of incomplete laps to maintain high quality.

***

## Model Overview

- **LSTM Neural Network:**  
  A sequential model that learns from past lap telemetry features (using sliding windows of 3 laps) to predict the next lap’s total time.
- **Random Forest Regressor:**  
  A classic ML baseline that uses flattened sequences of lap features for comparison.
- Both models are trained on 80% of the data and validated on the remaining 20%.

***

## How to Run

1. Ensure you have Python 3.8+ installed.
2. Install necessary dependencies:
   ```
   pip install fastf1 tensorflow scikit-learn pandas matplotlib
   ```
3. Run the Python script or Jupyter notebook containing the project code.
4. The script will automatically download telemetry data, preprocess, train models, and produce evaluation metrics alongside plots.

***

## Results

- Metrics including RMSE, MAE, and R² score are printed to evaluate prediction performance.
- Visualizations show actual vs predicted lap times from both LSTM and Random Forest models over the test set.
- The LSTM model consistently outperforms the Random Forest baseline, demonstrating the power of sequential deep learning in modeling race telemetry.

***

## Potential Improvements

- Incorporate additional telemetry inputs (e.g., fuel load, weather conditions, driver input data).
- Expand dataset with more drivers and race events for generalization.
- Hyperparameter tuning and architecture search to improve deep learning model performance.
- Deploy model in a real-time prediction setting with live telemetry streams.

***

## Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) for providing the official F1 telemetry API.
- TensorFlow and scikit-learn libraries for machine learning functionality.
- Formula 1 for the publicly available race data.

***

## Contact

For questions or suggestions, please reach out via GitHub issues or contact me at `garresatvik@gmail.com`.

***
