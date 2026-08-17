# hydro_climate_LSTM_Mumbai
I built a hydro-climatology ML model for Mumbai using 2010-2025 NASA POWER data. Lacking discharge records, I used the SCS Curve Number method to create a runoff proxy. I trained an LSTM on 30-day hydro-met features to predict next-day runoff events, achieving 0.965 ROC-AUC and 0.736 F1-score.
