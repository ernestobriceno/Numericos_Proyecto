import gradio as gr
import numpy as np
import pandas as pd
import joblib
import json

# ==============================================================================
# FUNCIONES DEL MODELO DE REGRESIÓN LOGÍSTICA 
# ==============================================================================
def funcion_sigmoide(z):
    z_clipped = np.clip(z, -500, 500) # Prevenir overflow/underflow
    g = 1 / (1 + np.exp(-z_clipped))
    return g

def funcion_hipotesis_logistica(X_matrix_b, theta_vector):
    if theta_vector.ndim == 1: # Asegurar que theta es un vector columna
        theta_vector = theta_vector.reshape(-1, 1)
    z = X_matrix_b @ theta_vector # Multiplicación de matrices
    return funcion_sigmoide(z)

# ==============================================================================
# CARGA DE ARTEFACTOS DE LOS MODELOS
# ==============================================================================

# --- Modelo de Regresión Logística (Donación de Sangre) ---
try:
    theta_logistic = np.load('theta_logistic.npy')
    scaler_logistic = joblib.load('scaler_logistic.joblib')
    with open('feature_names_logistic.json', 'r') as f:
        feature_names_logistic = json.load(f)
    print("Artefactos del modelo logístico cargados exitosamente.")
except FileNotFoundError:
    print("Error: No se encontraron 'theta_logistic.npy', 'scaler_logistic.joblib' o 'feature_names_logistic.json'.")
    print("Nota: desde el notebook el modelo ya entrenado")
    # Fallback para que la app no crashee al inicio, pero las predicciones fallarán.
    theta_logistic = None 
    scaler_logistic = None
    feature_names_logistic = ['Recency_months', 'Frequency_times', 'Time_months'] # Placeholder
except Exception as e:
    print(f"Error al cargar artefactos logísticos: {e}")
    theta_logistic = None
    scaler_logistic = None
    feature_names_logistic = ['Recency_months', 'Frequency_times', 'Time_months']
