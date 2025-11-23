import numpy as np
import scipy.signal
import json
from typing import List, Dict, Any


def get_maintenance_windows(risk_input_list: List[float]) -> Dict[str, Any]:
    """
    Calcula ventanas óptimas de mantenimiento basadas en riesgos de sensores.
    
    Input: Lista de riesgos actuales de los sensores [0.4, 0.5, 0.2...]
    Output: Dict con la curva de costos y los puntos óptimos de mantenimiento.
    
    Args:
        risk_input_list: Lista de valores de riesgo (0-1) de diferentes sensores
        
    Returns:
        Dictionary con:
        - series_data: Puntos de la curva de costo total
        - annotations: Puntos óptimos para realizar mantenimiento
    """
    
    # 1. Procesar el Input
    # El riesgo promedio actual define qué tan "alta" empieza la curva
    avg_risk_factor = np.mean(risk_input_list) if risk_input_list else 0.5
    
    # 2. Generar Datos Temporales (0 a 36 meses)
    months = np.linspace(0, 36, 100)
    
    # 3. Función de Costo Total (Simulación de "Valles" de oportunidad)
    # Costo Base (Sube por degradación natural) + Factor de Riesgo Input
    base_curve = 10000 * np.exp(0.08 * months) * (0.5 + avg_risk_factor)
    
    # Creamos "Ventanas de Oportunidad" (Valles artificiales donde baja el costo operativo)
    # Esto simula disponibilidad de maquinaria, clima favorable, etc.
    seasonality = -5000 * np.sin(months * 0.5) - 3000 * np.cos(months * 0.2)
    
    total_cost_curve = base_curve + seasonality
    
    # 4. Encontrar los Mínimos Locales (Los 3 puntos óptimos)
    # Usamos argrelextrema para encontrar los valles
    minima_indices = scipy.signal.argrelextrema(total_cost_curve, np.less)[0]
    
    # Tomamos hasta 3 mínimos, ordenados por fecha
    optimal_points = []
    for idx in minima_indices[:3]:
        optimal_points.append({
            "month": round(float(months[idx]), 1),
            "cost": round(float(total_cost_curve[idx]), 2),
            "label": f"Mes {int(months[idx])}"
        })
        
    # 5. Formatear para Frontend (ApexCharts)
    # Reducimos la cantidad de puntos para no saturar el JSON
    chart_data = []
    for m, c in zip(months[::2], total_cost_curve[::2]):  # Downsampling
        chart_data.append({"x": round(float(m), 1), "y": round(float(c), 2)})
        
    response = {
        "series_data": chart_data,  # La línea
        "annotations": optimal_points,  # Los marcadores
        "avg_risk_factor": round(float(avg_risk_factor), 3),
        "total_points": len(risk_input_list)
    }
    
    return response


def job_maintain_service(critical_points: List[float]) -> Dict[str, Any]:
    """
    Servicio principal que procesa puntos críticos y retorna ventanas de mantenimiento.
    
    Args:
        critical_points: Lista de valores de riesgo de puntos críticos
        
    Returns:
        Dictionary con los datos de las ventanas de mantenimiento optimizadas
    """
    if not critical_points:
        return {
            "error": "No se proporcionaron puntos críticos",
            "series_data": [],
            "annotations": []
        }
    
    # Validar que todos los valores estén en el rango esperado
    valid_points = [p for p in critical_points if 0 <= p <= 1]
    
    if len(valid_points) != len(critical_points):
        print(f"Advertencia: Se filtraron {len(critical_points) - len(valid_points)} puntos fuera del rango [0, 1]")
    
    if not valid_points:
        return {
            "error": "No hay puntos críticos válidos (deben estar entre 0 y 1)",
            "series_data": [],
            "annotations": []
        }
    
    # Calcular las ventanas de mantenimiento
    maintenance_windows = get_maintenance_windows(valid_points)
    
    return maintenance_windows
