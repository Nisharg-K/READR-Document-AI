"""
GPU-Accelerated OCR service for FastAPI backend.
Integrates EasyOCR with CUDA support for fast PDF processing.
"""

import asyncio
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import easyocr
import fitz  # PyMuPDF

warnings.filterwarnings("ignore")


class GPUOCRService:
    """Singleton service for GPU-accelerated OCR."""

    _instance = None
    _reader = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize OCR service on first creation."""
        if self._reader is None:
            self._init_reader()

    def _init_reader(self):
        """Initialize EasyOCR reader with GPU support."""
        use_gpu = torch.cuda.is_available()

        if use_gpu:
            print(f"✓ CUDA Available: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU Memory: {memory_gb:.2f} GB")

        print("Loading EasyOCR model (GPU-accelerated)...")
        self._reader = easyocr.Reader(
            ["en"],
            gpu=use_gpu,
            quantize=True
        )
        print("✓ OCR model ready\n")

    def extract_text_from_pdf(
        self,
        pdf_path: str,
        zoom: float = 2.0,
        max_pages: Optional[int] = None
    ) -> dict:
        """
        Extract text from PDF using GPU-accelerated OCR.

        Args:
            pdf_path: Path to PDF file
            zoom: Rendering quality (1.0 = 72 DPI, 2.0 = 144 DPI)
            max_pages: Max pages to process (None = all)

        Returns:
            Dictionary with extracted text and metadata
        """
        pdf_doc = fitz.open(pdf_path)
        num_pages = len(pdf_doc)

        if max_pages:
            num_pages = min(num_pages, max_pages)

        extracted_data = {
            "filename": Path(pdf_path).name,
            "total_pages": len(pdf_doc),
            "processed_pages": num_pages,
            "pages": []
        }

        for page_num in range(num_pages):
            page = pdf_doc[page_num]

            # Render page to high-quality image
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Convert to numpy array (RGB)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8)
            img_array = img_array.reshape((pix.height, pix.width, pix.n))

            # Run OCR
            results = self._reader.readtext(img_array, batch_size=1, workers=0)

            # Extract text and confidence
            page_text = "\n".join([item[1] for item in results])
            avg_confidence = np.mean([item[2] for item in results]) if results else 0

            extracted_data["pages"].append({
                "page_num": page_num + 1,
                "text": page_text,
                "confidence": float(avg_confidence),
                "text_length": len(page_text)
            })

        pdf_doc.close()

        # Calculate summary stats
        total_text = "\n\n".join([p["text"] for p in extracted_data["pages"]])
        extracted_data["total_text_length"] = len(total_text)
        extracted_data["avg_confidence"] = float(
            np.mean([p["confidence"] for p in extracted_data["pages"]])
        )

        return extracted_data

    def extract_text_from_image(self, image_path: str) -> dict:
        """
        Extract text from image using GPU-accelerated OCR.

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with extracted text and metadata
        """
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Could not load image: {image_path}"}

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self._reader.readtext(image, batch_size=1, workers=4)

        text = "\n".join([item[1] for item in results])
        confidence = np.mean([item[2] for item in results]) if results else 0

        return {
            "filename": Path(image_path).name,
            "text": text,
            "confidence": float(confidence),
            "detections": len(results)
        }


# Global service instance
ocr_service = GPUOCRService()
