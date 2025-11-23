from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from inference_service import inference_service
from job_maintain_service import job_maintain_service
from typing import List

app = FastAPI()

# ⭐ Configuración de CORS - ESTO ES LO QUE NECESITAS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Puerto por defecto de Vite
        "http://localhost:3000",  # Por si usas otro puerto
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Permite todos los headers
)



@app.get("/api/inference")
async def inference():
    return inference_service()


@app.post("/api/maintenance-windows")
async def maintenance_windows(critical_points: List[float] = Body(..., embed=True)):
    """
    Endpoint para calcular ventanas óptimas de mantenimiento.
    
    Body esperado:
    {
        "critical_points": [0.2, 0.3, 0.25, 0.4, 0.5]
    }
    
    Returns:
        JSON con series_data (curva de costos) y annotations (puntos óptimos)
    """
    return job_maintain_service(critical_points)


@app.get("/")
async def root():
    return {"message": "Hello World"}
