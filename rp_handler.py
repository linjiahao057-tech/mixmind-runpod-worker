import os
import base64
import tempfile
import traceback

import runpod
import torch
from qwen_asr import Qwen3ASRModel

ASR_MODEL = os.environ.get("ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
ALIGNER_MODEL = os.environ.get("ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")

# Load once per RunPod worker. The forced aligner returns word/character timestamps.
model = Qwen3ASRModel.from_pretrained(
    ASR_MODEL,
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=1,
    max_new_tokens=4096,
    forced_aligner=ALIGNER_MODEL,
    forced_aligner_kwargs=dict(
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ),
)


def _decode_audio(value):
    raw = str(value or "")
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    data = base64.b64decode(raw)
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.write(data)
    handle.close()
    return handle.name


def _timestamp_rows(result):
    """Normalize qwen-asr timestamp objects into plain JSON rows."""
    stamps = getattr(result, "time_stamps", None)
    if stamps is None and isinstance(result, dict):
        stamps = result.get("time_stamps")
    if not stamps:
        return []

    # qwen-asr may return one list of aligned items per audio input.
    groups = stamps if isinstance(stamps, (list, tuple)) else [stamps]
    if groups and isinstance(groups[0], (list, tuple)):
        items = groups[0]
    else:
        items = groups

    rows = []
    for item in items:
        text = getattr(item, "text", None)
        start = getattr(item, "start_time", None)
        end = getattr(item, "end_time", None)
        if isinstance(item, dict):
            text = item.get("text", text)
            start = item.get("start_time", start)
            end = item.get("end_time", end)
        if text is None or start is None or end is None:
            continue
        rows.append({
            "text": str(text),
            "startMs": round(float(start) * 1000),
            "endMs": round(float(end) * 1000),
        })
    return rows


def handler(job):
    temp_path = None
    try:
        job_input = job.get("input") or {}
        audio_base64 = job_input.get("audio_base64")
        if not audio_base64:
            return {"error": "missing input.audio_base64"}

        temp_path = _decode_audio(audio_base64)
        language = job_input.get("language") or None
        results = model.transcribe(
            audio=temp_path,
            language=language,
            return_time_stamps=True,
        )
        result = results[0]
        words = _timestamp_rows(result)
        text = getattr(result, "text", None)
        if text is None and isinstance(result, dict):
            text = result.get("text", "")
        return {
            "text": str(text or ""),
            "language": getattr(result, "language", None) or language or "auto",
            "words": words,
            "word_count": len(words),
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "traceback": traceback.format_exc(limit=8),
        }
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
