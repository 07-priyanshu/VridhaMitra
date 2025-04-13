from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import shutil
from app.data import predict_output as op  # Make sure this import is correct

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration - using pathlib for cross-platform compatibility
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Create upload directory if it doesn't exist
UPLOAD_DIR.mkdir(exist_ok=True)

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def cleanup_old_files(directory: Path, max_files: int = 20):
    """Clean up old files to prevent storage bloat"""
    files = sorted(directory.glob('*'), key=os.path.getmtime)
    while len(files) > max_files:
        try:
            files[0].unlink()
            files = files[1:]
        except Exception as e:
            print(f"Error cleaning up files: {e}")
            break

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.content_type.startswith('image/') or not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Only image files (JPEG, PNG, GIF) are allowed")

        # Validate file size
        temp_path = UPLOAD_DIR / f"temp_{os.urandom(8).hex()}"
        file_size = 0
        
        try:
            with open(temp_path, "wb") as buffer:
                while content := await file.read(1024 * 1024):  # Read in 1MB chunks
                    file_size += len(content)
                    if file_size > MAX_FILE_SIZE:
                        raise HTTPException(status_code=413, detail="File too large (max 10MB)")
                    buffer.write(content)
        finally:
            file.file.close()

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_ext = Path(file.filename).suffix
        saved_filename = f"img_{timestamp}{file_ext}"
        saved_path = UPLOAD_DIR / saved_filename

        # Rename temp file to final name
        os.rename(temp_path, saved_path)

        # Clean up old files
        cleanup_old_files(UPLOAD_DIR)

        # Process image
        key_enco_image, pose = op(str(saved_path))  # Convert Path to string for compatibility

        if not key_enco_image:
            raise HTTPException(status_code=422, detail=pose) 

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "filename": saved_filename,
                "path": str(saved_path),
                "size": file_size,
                "content_type": file.content_type,
                "Pose_Name": pose,
                "annotated_image": key_enco_image
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Ensure temp file is cleaned up if something went wrong
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
