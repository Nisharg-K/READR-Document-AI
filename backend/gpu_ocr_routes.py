"""
FastAPI endpoints for GPU-accelerated OCR.
Add these routes to your server.py file.

Usage:
    from fastapi import APIRouter
    from gpu_ocr_routes import router
    
    app.include_router(router)
"""

from pathlib import Path
import tempfile
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from gpu_ocr import ocr_service

router = APIRouter(prefix="/api/ocr", tags=["OCR"])


@router.post("/extract-from-pdf")
async def extract_from_pdf(
    file: UploadFile = File(...),
    zoom: float = Form(2.0),
    max_pages: int = Form(None)
):
    """
    Extract text from uploaded PDF using GPU-accelerated OCR.

    Args:
        file: PDF file to process
        zoom: Rendering quality (1.0 = standard, 2.0 = high quality)
        max_pages: Maximum pages to process (None = all)

    Returns:
        Extracted text with per-page metadata
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Extract text using GPU OCR
        result = ocr_service.extract_text_from_pdf(
            tmp_path,
            zoom=zoom,
            max_pages=max_pages
        )

        # Cleanup
        Path(tmp_path).unlink()

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-from-image")
async def extract_from_image(file: UploadFile = File(...)):
    """
    Extract text from uploaded image using GPU-accelerated OCR.

    Args:
        file: Image file (jpg, png, etc.)

    Returns:
        Extracted text with confidence scores
    """
    allowed_types = {'image/jpeg', 'image/png', 'image/tiff', 'image/gif'}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="File must be an image (jpg, png, tiff, gif)"
        )

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Extract text using GPU OCR
        result = ocr_service.extract_text_from_image(tmp_path)

        # Cleanup
        Path(tmp_path).unlink()

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Check OCR service status and GPU availability."""
    import torch
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "service": "GPU-accelerated OCR ready"
    }
