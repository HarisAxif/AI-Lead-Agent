"""Optional live Ollama smoke check; no web discovery or hosted inference."""
import base64
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import struct
import zlib
from pydantic import BaseModel
from local_llm import ensure_ready, structured_call, OLLAMA_MODEL
from clarify import validate_service_fit
from schemas import SearchRequest

class ImageCheck(BaseModel):
    dominant_color: str

print("Ready:", OLLAMA_MODEL, ensure_ready().get("capabilities"), flush=True)
request = SearchRequest(industry="local bakeries", location="Faisalabad",
                        service="quantum computing consulting", number_of_leads=10)
fit = validate_service_fit(request)
print("Bakery quantum fit:", fit.model_dump(), flush=True)
assert not fit.valid, "Mismatched service was accepted"
def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
width = height = 180
pixels = b"".join(b"\x00" + bytes([255, 0, 0]) * width for _ in range(height))
image = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
image += chunk(b"IDAT", zlib.compress(pixels)) + chunk(b"IEND", b"")
result = structured_call("Name the dominant color of this image.", ImageCheck,
                         images=[base64.b64encode(image).decode()])
print("Vision:", result.model_dump(), flush=True)
assert "red" in result.dominant_color.lower()
print("Local text and image inference passed.", flush=True)
