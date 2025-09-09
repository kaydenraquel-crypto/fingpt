"""
RunPod Optimized FinGPT Remote Server
====================================

Enhanced version of fingpt_remote_server.py optimized for RunPod deployment.
Includes monitoring, security, and cost optimization features.
"""

import asyncio
import json
import logging
import os
import sys
import time
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psutil
import torch

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fingpt_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RunPodFinGPT")

# Configuration
API_KEY = os.getenv('FINGPT_API_KEY', 'runpod-fingpt-2024')
IDLE_SHUTDOWN_MINUTES = int(os.getenv('IDLE_SHUTDOWN_MINUTES', '30'))
MAX_BATCH_SIZE = int(os.getenv('MAX_BATCH_SIZE', '4'))
ENABLE_AUTO_SHUTDOWN = os.getenv('ENABLE_AUTO_SHUTDOWN', 'true').lower() == 'true'

# Security
security = HTTPBearer(auto_error=False)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key for secure access"""
    if not credentials:
        raise HTTPException(status_code=401, detail="API key required")
    
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Log request for idle tracking
    with open('/tmp/last_request', 'w') as f:
        f.write(str(time.time()))
    
    return credentials

# Request/Response Models
class AnalysisRequest(BaseModel):
    query: str
    portfolio_value: float = 10000
    max_tokens: int = 512
    temperature: float = 0.7

class AnalysisResponse(BaseModel):
    success: bool
    recommendation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: int = 0
    server_info: Dict[str, Any] = {}

class SystemStats(BaseModel):
    gpu_utilization: float
    gpu_memory_used: int
    gpu_memory_total: int
    gpu_temperature: float
    cpu_percent: float
    ram_percent: float
    uptime_hours: float

# Global variables
app = FastAPI(
    title="RunPod FinGPT Server",
    version="2.0.0",
    description="GPU-accelerated FinGPT server optimized for RunPod deployment"
)

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fingpt_advisor = None
server_start_time = time.time()

def get_gpu_stats() -> Dict[str, Any]:
    """Get detailed GPU statistics"""
    try:
        if not torch.cuda.is_available():
            return {
                "available": False,
                "error": "CUDA not available"
            }
        
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        
        # Get detailed stats via nvidia-smi
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            stats = result.stdout.strip().split(', ')
            return {
                "available": True,
                "name": gpu_name,
                "count": gpu_count,
                "utilization_percent": float(stats[0]) if stats[0] != '[Not Supported]' else 0,
                "memory_used_mb": int(stats[1]),
                "memory_total_mb": int(stats[2]),
                "temperature_c": float(stats[3]) if stats[3] != '[Not Supported]' else 0,
                "power_draw_w": float(stats[4]) if len(stats) > 4 and stats[4] != '[Not Supported]' else 0
            }
        else:
            return {
                "available": True,
                "name": gpu_name,
                "count": gpu_count,
                "error": "nvidia-smi unavailable"
            }
            
    except Exception as e:
        logger.error(f"Error getting GPU stats: {e}")
        return {
            "available": False,
            "error": str(e)
        }

def get_system_stats() -> SystemStats:
    """Get comprehensive system statistics"""
    gpu_stats = get_gpu_stats()
    uptime = time.time() - server_start_time
    
    return SystemStats(
        gpu_utilization=gpu_stats.get('utilization_percent', 0),
        gpu_memory_used=gpu_stats.get('memory_used_mb', 0),
        gpu_memory_total=gpu_stats.get('memory_total_mb', 0),
        gpu_temperature=gpu_stats.get('temperature_c', 0),
        cpu_percent=psutil.cpu_percent(),
        ram_percent=psutil.virtual_memory().percent,
        uptime_hours=uptime / 3600
    )

async def initialize_fingpt():
    """Initialize FinGPT advisor"""
    global fingpt_advisor
    
    try:
        logger.info("🚀 Initializing FinGPT advisor...")
        
        # Add FinGPT to path (adjust path as needed)
        fingpt_path = "/workspace/FinGPT"
        if os.path.exists(fingpt_path):
            sys.path.append(fingpt_path)
        
        # Try multiple import paths
        try:
            from fingpt.FinGPT_Forecaster import FinGPTForecaster
            fingpt_advisor = FinGPTForecaster()
        except ImportError:
            try:
                # Alternative import
                sys.path.append("/workspace")
                from SuperNovaAdvisor import SuperNovaAdvisor
                fingpt_advisor = SuperNovaAdvisor()
            except ImportError:
                # Mock advisor for testing
                logger.warning("⚠️ FinGPT not found, using mock advisor")
                fingpt_advisor = MockFinGPTAdvisor()
        
        logger.info("✅ FinGPT advisor initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize FinGPT: {e}")
        return False

class MockFinGPTAdvisor:
    """Mock advisor for testing when FinGPT is not available"""
    
    async def analyze_request(self, query: str, portfolio_value: float):
        """Mock analysis for testing"""
        await asyncio.sleep(0.5)  # Simulate processing time
        
        return {
            "action": "BUY",
            "confidence": 0.75,
            "symbol": "AAPL", 
            "asset_class": "stocks",
            "entry_price": 150.00,
            "target_price": 165.00,
            "stop_loss": 140.00,
            "position_size_percent": 5.0,
            "probability_of_success": 0.68,
            "expected_return": 10.0,
            "max_drawdown_risk": -6.7,
            "holding_period": "3-6 months",
            "rationale": f"Mock analysis for: {query}. This is a test response from RunPod FinGPT server.",
            "key_factors": ["Technical analysis", "Market sentiment", "Fundamental strength"],
            "risk_warnings": ["Market volatility", "Economic uncertainty"],
            "market_context": "Current market conditions favor growth stocks",
            "timestamp": datetime.now().isoformat(),
            "analysis_duration_ms": 500,
            "source": "mock_fingpt"
        }

# Background task for idle shutdown
async def idle_shutdown_monitor():
    """Monitor for idle periods and auto-shutdown to save costs"""
    if not ENABLE_AUTO_SHUTDOWN:
        return
    
    logger.info(f"🕐 Auto-shutdown enabled: {IDLE_SHUTDOWN_MINUTES} minutes idle timeout")
    
    while True:
        try:
            # Check last request time
            if os.path.exists('/tmp/last_request'):
                with open('/tmp/last_request', 'r') as f:
                    last_request = float(f.read().strip())
                
                idle_minutes = (time.time() - last_request) / 60
                
                if idle_minutes > IDLE_SHUTDOWN_MINUTES:
                    logger.info(f"🛑 Auto-shutdown: {idle_minutes:.1f} minutes idle")
                    logger.info("💰 Shutting down to save costs...")
                    
                    # Graceful shutdown
                    os.system("shutdown -h +1")  # Shutdown in 1 minute
                    break
                else:
                    logger.debug(f"⏱️ Idle time: {idle_minutes:.1f}/{IDLE_SHUTDOWN_MINUTES} minutes")
            
            # Check every 5 minutes
            await asyncio.sleep(300)
            
        except Exception as e:
            logger.error(f"❌ Idle monitor error: {e}")
            await asyncio.sleep(60)

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize server on startup"""
    logger.info("🌟 RunPod FinGPT Server starting...")
    
    # Log system info
    gpu_stats = get_gpu_stats()
    logger.info(f"🎮 GPU: {gpu_stats.get('name', 'Unknown')}")
    logger.info(f"💾 GPU Memory: {gpu_stats.get('memory_total_mb', 0)} MB")
    logger.info(f"🔑 API Key: {'✅ Set' if API_KEY else '❌ Not Set'}")
    
    # Initialize FinGPT
    success = await initialize_fingpt()
    if not success:
        logger.warning("⚠️ FinGPT initialization failed, server will use fallback")
    
    # Start idle monitor
    if ENABLE_AUTO_SHUTDOWN:
        asyncio.create_task(idle_shutdown_monitor())
    
    # Set initial request time
    with open('/tmp/last_request', 'w') as f:
        f.write(str(time.time()))
    
    logger.info("🚀 RunPod FinGPT Server ready!")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_stats = get_gpu_stats()
    system_stats = get_system_stats()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_hours": system_stats.uptime_hours,
        "gpu_available": gpu_stats.get('available', False),
        "gpu_name": gpu_stats.get('name', 'Unknown'),
        "fingpt_ready": fingpt_advisor is not None,
        "version": "2.0.0",
        "environment": "runpod"
    }

@app.get("/stats")
async def get_stats(credentials: HTTPAuthorizationCredentials = Depends(verify_api_key)):
    """Get detailed system statistics"""
    return get_system_stats()

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_query(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(verify_api_key)
):
    """Analyze trading query using FinGPT"""
    start_time = time.time()
    
    try:
        logger.info(f"📝 Processing query: {request.query[:50]}...")
        
        if not fingpt_advisor:
            raise HTTPException(status_code=503, detail="FinGPT advisor not available")
        
        # Process the request
        if hasattr(fingpt_advisor, 'analyze_request'):
            # Standard SuperNova interface
            recommendation = await fingpt_advisor.analyze_request(
                request.query, 
                request.portfolio_value
            )
            
            # Convert to dictionary if needed
            if hasattr(recommendation, '__dict__'):
                rec_dict = recommendation.__dict__
            else:
                rec_dict = recommendation
                
        else:
            # Mock advisor interface
            rec_dict = await fingpt_advisor.analyze_request(
                request.query,
                request.portfolio_value
            )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Get server info
        gpu_stats = get_gpu_stats()
        server_info = {
            "processing_time_ms": processing_time,
            "gpu_name": gpu_stats.get('name', 'Unknown'),
            "gpu_utilization": gpu_stats.get('utilization_percent', 0),
            "server_version": "2.0.0",
            "environment": "runpod"
        }
        
        logger.info(f"✅ Analysis completed in {processing_time}ms")
        
        return AnalysisResponse(
            success=True,
            recommendation=rec_dict,
            processing_time_ms=processing_time,
            server_info=server_info
        )
        
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        logger.error(f"❌ Analysis failed: {e}")
        
        return AnalysisResponse(
            success=False,
            error=str(e),
            processing_time_ms=processing_time,
            server_info={
                "error_type": type(e).__name__,
                "server_version": "2.0.0",
                "environment": "runpod"
            }
        )

@app.post("/shutdown")
async def shutdown_server(credentials: HTTPAuthorizationCredentials = Depends(verify_api_key)):
    """Manually shutdown server (for cost savings)"""
    logger.info("🛑 Manual shutdown requested")
    
    # Schedule shutdown
    asyncio.create_task(delayed_shutdown())
    
    return {
        "message": "Server shutting down in 30 seconds",
        "timestamp": datetime.now().isoformat()
    }

async def delayed_shutdown():
    """Delayed shutdown to allow response to be sent"""
    await asyncio.sleep(30)
    os.system("shutdown -h now")

@app.get("/")
async def root():
    """Root endpoint with server info"""
    return {
        "name": "RunPod FinGPT Server",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze (POST, requires API key)",
            "stats": "/stats (GET, requires API key)",
            "shutdown": "/shutdown (POST, requires API key)"
        },
        "documentation": "/docs"
    }

if __name__ == "__main__":
    # Log startup info
    print("🚀 Starting RunPod FinGPT Server...")
    print(f"🔑 API Key: {'✅ Set' if API_KEY else '❌ Not Set'}")
    print(f"🕐 Auto-shutdown: {'✅ Enabled' if ENABLE_AUTO_SHUTDOWN else '❌ Disabled'} ({IDLE_SHUTDOWN_MINUTES}min)")
    print(f"📊 Visit http://localhost:8003/docs for API documentation")
    print(f"🏥 Health check: http://localhost:8003/health")
    
    # Start server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info",
        access_log=True
    )
