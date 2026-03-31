"""
GPU OCR Integration with READR Server
======================================

This file demonstrates how to use the GPU OCR and Ollama VLM features
integrated into the READR backend server.
"""

import requests
from pathlib import Path

# Server configuration
SERVER_URL = "http://localhost:8000"

def check_ocr_capabilities():
    """Check what OCR methods are available on the server."""
    response = requests.get(f"{SERVER_URL}/models")
    data = response.json()
    
    print("OCR Capabilities:")
    print(f"  GPU OCR Available: {data['ocr_methods']['gpu']}")
    print(f"  Ollama VLM Available: {data['ocr_methods']['ollama']}")
    
    if data['gpu_info']['cuda_available']:
        print(f"\nGPU Information:")
        print(f"  GPU: {data['gpu_info']['gpu_name']}")
        print(f"  Memory: {data['gpu_info']['gpu_memory_gb']} GB")
        print(f"  CUDA Version: {data['gpu_info']['cuda_version']}")
    else:
        print("\n⚠ GPU not available - will use CPU for processing")
    
    print(f"\nOllama Vision Model: {data['ocr_model']}")
    print(f"PyTorch Version: {data['gpu_info']['pytorch_version']}")


def ocr_pdf_gpu(pdf_path: str) -> dict:
    """Extract text from PDF using GPU-accelerated OCR."""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    with open(pdf_path, "rb") as f:
        files = {"file": (Path(pdf_path).name, f)}
        data = {"ocr_method": "gpu"}
        response = requests.post(f"{SERVER_URL}/ocr", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ OCR Complete (GPU)")
        print(f"  Method: {result['ocr_method']}")
        print(f"  Text Length: {result['text_length']} characters")
        print(f"  Extracted Text:\n{result['text'][:500]}...")
        return result
    else:
        print(f"❌ OCR Failed: {response.text}")
        return None


def ocr_pdf_ollama(pdf_path: str) -> dict:
    """Extract text from PDF using Ollama Vision Model."""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    with open(pdf_path, "rb") as f:
        files = {"file": (Path(pdf_path).name, f)}
        data = {"ocr_method": "ollama"}
        response = requests.post(f"{SERVER_URL}/ocr", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ OCR Complete (Ollama VLM)")
        print(f"  Method: {result['ocr_method']}")
        print(f"  Text Length: {result['text_length']} characters")
        print(f"  Extracted Text:\n{result['text'][:500]}...")
        return result
    else:
        print(f"❌ OCR Failed: {response.text}")
        return None


def ocr_pdf_auto(pdf_path: str) -> dict:
    """Extract text from PDF using automatic OCR method selection."""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    with open(pdf_path, "rb") as f:
        files = {"file": (Path(pdf_path).name, f)}
        data = {"ocr_method": "auto"}  # Auto-selects GPU if available, falls back to Ollama
        response = requests.post(f"{SERVER_URL}/ocr", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ OCR Complete (Auto)")
        print(f"  Method Used: {result['ocr_method']}")
        print(f"  Text Length: {result['text_length']} characters")
        print(f"  Extracted Text:\n{result['text'][:500]}...")
        return result
    else:
        print(f"❌ OCR Failed: {response.text}")
        return None


def upload_document_gpu(pdf_path: str, model_name: str = None) -> dict:
    """Upload PDF and process with GPU OCR, then store in vector DB."""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    with open(pdf_path, "rb") as f:
        files = {"file": (Path(pdf_path).name, f)}
        data = {
            "ocr_method": "gpu",
            "model_name": model_name or ""
        }
        response = requests.post(f"{SERVER_URL}/upload", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Document Uploaded & Indexed")
        print(f"  Doc ID: {result['doc_id']}")
        print(f"  Word Count: {result['word_count']}")
        print(f"  Chunk Count: {result['chunk_count']}")
        print(f"  Summary: {result['summary'][:200]}...")
        print(f"  Model Used: {result['model_name']}")
        return result
    else:
        print(f"❌ Upload Failed: {response.text}")
        return None


def upload_document_ollama(pdf_path: str, model_name: str = None) -> dict:
    """Upload PDF and process with Ollama VLM, then store in vector DB."""
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    with open(pdf_path, "rb") as f:
        files = {"file": (Path(pdf_path).name, f)}
        data = {
            "ocr_method": "ollama",
            "model_name": model_name or ""
        }
        response = requests.post(f"{SERVER_URL}/upload", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Document Uploaded & Indexed")
        print(f"  Doc ID: {result['doc_id']}")
        print(f"  Word Count: {result['word_count']}")
        print(f"  Chunk Count: {result['chunk_count']}")
        print(f"  Summary: {result['summary'][:200]}...")
        print(f"  Model Used: {result['model_name']}")
        return result
    else:
        print(f"❌ Upload Failed: {response.text}")
        return None


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("READR GPU OCR Integration Examples")
    print("=" * 70)
    
    # Check server capabilities
    print("\n1. Checking OCR Capabilities...")
    check_ocr_capabilities()
    
    # Example: Process a PDF with different methods
    # Uncomment and update the path to test with your PDF
    
    # pdf_file = "path/to/your/document.pdf"
    # 
    # print("\n2. Testing GPU OCR...")
    # try:
    #     ocr_pdf_gpu(pdf_file)
    # except Exception as e:
    #     print(f"Error: {e}")
    # 
    # print("\n3. Testing Ollama VLM OCR...")
    # try:
    #     ocr_pdf_ollama(pdf_file)
    # except Exception as e:
    #     print(f"Error: {e}")
    # 
    # print("\n4. Testing Auto OCR (fastest available)...")
    # try:
    #     ocr_pdf_auto(pdf_file)
    # except Exception as e:
    #     print(f"Error: {e}")
    # 
    # print("\n5. Uploading and indexing with GPU OCR...")
    # try:
    #     upload_document_gpu(pdf_file)
    # except Exception as e:
    #     print(f"Error: {e}")


"""
API Endpoints & Usage:

1. GET /models
   - Get available models and OCR capabilities
   - Returns GPU info, OCR methods, and model details

2. POST /ocr
   - Extract text from a scanned PDF
   - Parameters:
     - file: PDF file (multipart)
     - ocr_method: "gpu", "ollama", or "auto"
   - Returns: extracted text, text length, method used

3. POST /upload
   - Upload PDF, perform OCR, and index in vector DB
   - Parameters:
     - file: PDF file (multipart)
     - model_name: LLM to use for analysis (optional)
     - ocr_method: "gpu", "ollama", or "auto"
   - Returns: doc_id, summary, entities, insights, recommendations

OCR Method Comparison:

GPU OCR (CUDA):
  ✓ 10-50x faster than CPU
  ✓ Offline processing (no network needed)
  ✓ Works with scanned PDFs
  ✗ Requires GPU with CUDA support
  ✗ Limited language support
  Performance: ~1-2 seconds per page (RTX 4090)

Ollama VLM:
  ✓ Works offline
  ✓ Better contextual understanding
  ✓ Supports multiple languages
  ✗ Slower than GPU OCR
  ✗ Requires Ollama running
  ✗ More resource intensive
  Performance: ~5-10 seconds per page

Auto Mode:
  - Automatically uses GPU OCR if available
  - Falls back to Ollama VLM if GPU not available
  - Best for compatibility and performance
"""
