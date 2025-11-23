from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference_service import inference_service
from job_maintain_service import job_maintain_service
from pydantic import BaseModel
from typing import List, Dict
from datetime import datetime, timedelta
from dotenv import load_dotenv
import httpx
import os

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Default Vite port
        "http://localhost:3000",  # Alternative port
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DeepSeek API Configuration (loaded from .env)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")

# Validate API key is configured
if not DEEPSEEK_API_KEY:
    raise ValueError(
        "DEEPSEEK_API_KEY not found in environment variables. "
        "Please create a .env file with your API key. "
        "See .env.example for reference."
    )


# Pydantic Models
class MaintenanceWindowRequest(BaseModel):
    critical_points: List[float]


class AIReportRequest(BaseModel):
    maintenance_data: Dict



@app.get("/api/inference")
async def inference():
    return inference_service()


@app.post("/api/maintenance-windows")
async def maintenance_windows(critical_points: List[float] = Body(..., embed=True)):
    """
    Endpoint to calculate optimal maintenance windows.
    
    Expected body:
    {
        "critical_points": [0.2, 0.3, 0.25, 0.4, 0.5]
    }
    
    Returns:
        JSON with series_data (cost curve) and annotations (optimal points)
    """
    return job_maintain_service(critical_points)


@app.post("/api/generate-report")
async def generate_ai_report(request: AIReportRequest):
    """
    Generate an AI-powered maintenance report using DeepSeek API.
    
    Expected body:
    {
        "maintenance_data": {
            "annotations": [...],
            "total_points": 5,
            "avg_risk_factor": 0.32
        }
    }
    
    Returns:
        JSON with AI-generated report and recommendations
    """
    try:
        maintenance_data = request.maintenance_data
        
        # Prepare maintenance windows with dates
        current_date = datetime.now()
        maintenance_windows = []
        for annotation in maintenance_data.get('annotations', []):
            month_delta = annotation['month']
            future_date = current_date + timedelta(days=30 * month_delta)
            maintenance_windows.append({
                'date': future_date.strftime('%B %Y'),
                'cost': annotation['cost'],
                'month_from_now': month_delta
            })
        
        # Create the prompt for DeepSeek (in English)
        prompt = f"""Based on the comprehensive analysis of railway infrastructure conditions, prepare a technical maintenance assessment report.

Technical Analysis Summary:
- Critical infrastructure points requiring attention: {maintenance_data.get('total_points', 0)}
- Calculated risk factor: {maintenance_data.get('avg_risk_factor', 0) * 100:.1f}%
- Identified optimal intervention periods: {len(maintenance_windows)}

Recommended Track Tamping Intervention Schedule:
{chr(10).join([f"- {w['date']}: Estimated operational cost of ${w['cost']:.2f} (scheduled for {w['month_from_now']} months ahead)" for w in maintenance_windows])}

Prepare a formal technical report that provides comprehensive justification for implementing track tamping maintenance operations at the identified kilometer points. The report must address:

1. Operational cost optimization strategies
2. Maintenance efficiency maximization protocols
3. Risk mitigation for critical infrastructure failures
4. Financial return on investment analysis

Report Requirements:
- Professional technical language appropriate for executive and engineering stakeholders
- Maximum length of 250 words
- Integration of both technical and financial perspectives
- Specific reference to the identified maintenance windows
- Clear explanation of optimal maintenance window methodology
- Standard railway maintenance terminology throughout
- Formal academic writing style without informal elements

Structure the report as continuous professional paragraphs suitable for inclusion in official maintenance planning documentation."""

        # Call DeepSeek API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                DEEPSEEK_API_URL,
                headers={
                    "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a senior railway infrastructure engineer and maintenance planning specialist with expertise in technical documentation and cost-benefit analysis. Provide formal, professional reports suitable for executive review and technical implementation."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 800
                }
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500, 
                    detail=f"DeepSeek API error: {response.status_code}"
                )
            
            result = response.json()
            ai_report = result['choices'][0]['message']['content']
            
            # Extract recommendations
            recommendations = []
            if maintenance_windows:
                recommendations.append(
                    f"Prioritize maintenance in {maintenance_windows[0]['date']}"
                )
                recommendations.append(
                    f"Budget approximately ${maintenance_windows[0]['cost']:.2f} for the first intervention"
                )
                if len(maintenance_windows) > 1:
                    recommendations.append(
                        f"Plan next intervention for {maintenance_windows[1]['date']}"
                    )
            
            return {
                "report": ai_report,
                "recommendations": recommendations,
                "maintenance_windows": maintenance_windows,
                "metadata": {
                    "generated_at": current_date.isoformat(),
                    "total_critical_points": maintenance_data.get('total_points', 0),
                    "avg_risk_factor": maintenance_data.get('avg_risk_factor', 0)
                }
            }
            
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504, 
            detail="Request to AI service timed out"
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Error connecting to AI service: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error generating report: {str(e)}"
        )


@app.get("/")
async def root():
    return {"message": "Hello World"}
