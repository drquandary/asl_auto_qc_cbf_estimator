from fastapi import FastAPI
from .routes import router


app = FastAPI(title="ASL Auto-QC and CBF Estimator")

# Include API routes under /api
app.include_router(router, prefix='/api')


@app.get('/')
async def root():
    """Root endpoint providing basic project information."""
    return {'detail': 'ASL Auto-QC and CBF Estimator API'}
