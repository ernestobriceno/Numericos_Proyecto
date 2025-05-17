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
# --- Modelo de Regresión Lineal (Ganancias de Empresas) ---
try:
    theta_lineal = np.load('theta_lineal.npy')
    mean_train_lineal = np.load('mean_train_lineal.npy')
    std_train_lineal = np.load('std_train_lineal.npy')
    with open('feature_names_lineal.json', 'r') as f:
        feature_names_lineal = json.load(f)
    print("Artefactos del modelo lineal (normalización manual) cargados exitosamente.")
except FileNotFoundError:
    print("Error: No se encontraron 'theta_lineal.npy', 'mean_train_lineal.npy', 'std_train_lineal.npy' o 'feature_names_lineal.json'.") 
    print("Nota: desde el notebook el modelo ya entrenado")
    theta_lineal = None
    mean_train_lineal, std_train_lineal, feature_names_lineal = None, None, None
except Exception as e:
    print(f"Error al cargar artefactos lineales: {e}")
    theta_lineal = None
    mean_train_lineal, std_train_lineal, feature_names_lineal = None, None, None
# ==============================================================================
# FUNCIONES DE PREDICCIÓN PARA GRADIO
# ==============================================================================

# --- Para Regresión Logística ---
def predecir_donacion_logistica(recencia, frecuencia, tiempo):
    if theta_logistic is None or scaler_logistic is None or feature_names_logistic is None:
        return "Error: Artefactos del modelo logístico no cargados. Verifica los archivos .npy, .joblib y .json."
    try:
        # 1. Crear DataFrame con las entradas del usuario en el orden correcto
        input_data = pd.DataFrame([[recencia, frecuencia, tiempo]], columns=feature_names_logistic)
        # 2. Escalar las características
        input_features_scaled = scaler_logistic.transform(input_data)
        # 3. Añadir el término de intercepto
        input_features_b = np.c_[np.ones((input_features_scaled.shape[0], 1)), input_features_scaled]
        # 4. Realizar la predicción de probabilidad
        probabilidad = funcion_hipotesis_logistica(input_features_b, theta_logistic)
        probabilidad_si_dona = probabilidad[0,0]
        # 5. Convertir probabilidad a clase
        clase_predicha = 1 if probabilidad_si_dona >= 0.5 else 0
        resultado_clase = "Sí Donará (Clase 1)" if clase_predicha == 1 else "No Donará (Clase 0)"
        return f"{resultado_clase}\nProbabilidad de que sí done: {probabilidad_si_dona:.4f}"
    except Exception as e:
        return f"Error durante la predicción logística: {str(e)}"

# --- Para Regresión Lineal ---
def predecir_profit_lineal(rd_spend, administration, marketing_spend, state_input):
    if theta_lineal is None or mean_train_lineal is None or std_train_lineal is None or feature_names_lineal is None:
        return "Error: Artefactos del modelo lineal no cargados. Verifica los archivos .npy y .json."
    try:
        # 1. Crear un diccionario para las entradas numéricas y las dummies de estado
        data_dict = {}
        data_dict['R&D Spend'] = rd_spend
        data_dict['Administration'] = administration
        data_dict['Marketing Spend'] = marketing_spend
        
        for feature_name in feature_names_lineal:
            if feature_name.startswith('State_'):
                data_dict[feature_name] = 0.0 # Inicializar todas las dummies de estado a 0
        
        # Activar la dummy correspondiente al estado seleccionado
        if state_input == "Florida" and 'State_Florida' in feature_names_lineal:
            data_dict['State_Florida'] = 1.0
        elif state_input == "New York" and 'State_New York' in feature_names_lineal:
            data_dict['State_New York'] = 1.0
        # 2. Crear un DataFrame con las entradas en el ORDEN EXACTO de feature_names_lineal
        input_df_lineal = pd.DataFrame([data_dict])[feature_names_lineal]
        
        # 3. Normalización manual Z-score
        input_features_lineal_normalized_values = (input_df_lineal.values - mean_train_lineal) / std_train_lineal
        
        # 4. Añadir el término de intercepto
        input_features_lineal_b = np.c_[np.ones((input_features_lineal_normalized_values.shape[0], 1)), input_features_lineal_normalized_values]
        
        # 5. Realizar la predicción de profit (función de hipótesis lineal)
        profit_predicho = (input_features_lineal_b @ theta_lineal)[0,0]
        
        return f"Profit Predicho (Lineal): ${profit_predicho:,.2f}"
        
    except Exception as e:
        return f"Error durante la predicción lineal: {str(e)}"
# ==============================================================================
# CREACIÓN DE LA INTERFAZ GRADIO CON PESTAÑAS
# ==============================================================================
with gr.Blocks(theme=gr.themes.Soft()) as demo_combinada:
    gr.Markdown("# Proyecto Análisis Numérico: Modelos Predictivos")
    gr.Markdown("Selecciona el modelo y proporciona las entradas para obtener una predicción.")

    with gr.Tabs():
        with gr.TabItem("🩸 Regresión Logística (Predicción de Donación de Sangre)"):
            gr.Markdown("### Ingresa los datos del donante:")
            with gr.Row():
                input_recencia = gr.Number(label="Recencia (meses desde última donación)", value=2)
                input_frecuencia = gr.Number(label="Frecuencia (número total de donaciones)", value=3)
                input_tiempo = gr.Number(label="Tiempo (meses desde primera donación)", value=10)
            
            output_logistica = gr.Textbox(label="Resultado de la Predicción (Logística)")
            btn_logistica = gr.Button("Predecir Donación")
            btn_logistica.click(
                predecir_donacion_logistica, 
                inputs=[input_recencia, input_frecuencia, input_tiempo], 
                outputs=output_logistica
            )
            gr.Examples(
                examples=[
                    [2, 3, 10], 
                    [2, 15, 30], 
                    [23, 1, 23] 
                ],
                inputs=[input_recencia, input_frecuencia, input_tiempo],
                outputs=output_logistica 
            )

        with gr.TabItem("📈 Regresión Lineal (Predicción de Profit de Empresas)"):
            gr.Markdown("### Ingresa los datos de la empresa:")
            with gr.Row():
                input_rd_spend = gr.Number(label="Gasto en I+D (R&D Spend)", value=100000)
                input_admin_spend = gr.Number(label="Gasto en Administración", value=120000)
            with gr.Row():
                input_marketing_spend = gr.Number(label="Gasto en Marketing", value=250000)
                input_estado_lineal = gr.Dropdown(
                    label="Estado de la Empresa", 
                    choices=["California", "New York", "Florida"], 
                    value="California" # Asume California como estado base si usaste drop_first
                )
            
            output_lineal = gr.Textbox(label="Resultado de la Predicción (Lineal)")
            btn_lineal = gr.Button("Predecir Profit")
            btn_lineal.click(
                predecir_profit_lineal, 
                inputs=[input_rd_spend, input_admin_spend, input_marketing_spend, input_estado_lineal], 
                outputs=output_lineal
            )
            gr.Examples(
                examples=[
                    [160000, 130000, 470000, "New York"],
                    [20000, 80000, 100000, "California"],
                    [75000, 100000, 200000, "Florida"]
                ],
                inputs=[input_rd_spend, input_admin_spend, input_marketing_spend, input_estado_lineal],
                outputs=output_lineal 
            )

# ==============================================================================
# LANZAMIENTO DE LA APLICACIÓN GRADIO
# ==============================================================================
if __name__ == "__main__":
    demo_combinada.launch(debug=True)
