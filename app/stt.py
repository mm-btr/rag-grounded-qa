"""ElevenLabs Scribe STT: one async multipart POST, no SDK."""
import aiohttp

from config import ELEVENLABS_API_KEY, STT_MODEL, STT_URL, STT_TIMEOUT


class STTError(RuntimeError):
    pass


def _parse_transcript(data):
    text = (data.get("text") or "").strip()
    if not text:
        raise STTError("пустой ответ распознавания")
    return text


async def transcribe(audio, filename="voice.oga"):
    if not ELEVENLABS_API_KEY:
        raise STTError("ELEVENLABS_API_KEY не задан")
    form = aiohttp.FormData()
    form.add_field("model_id", STT_MODEL)
    form.add_field("file", audio, filename=filename, content_type="application/octet-stream")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                STT_URL,
                data=form,
                headers={"xi-api-key": ELEVENLABS_API_KEY},
                timeout=aiohttp.ClientTimeout(total=STT_TIMEOUT),
            ) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:200]
                    raise STTError(f"Scribe {resp.status}: {body}")
                data = await resp.json()
    except aiohttp.ClientError as e:
        raise STTError(str(e)) from e
    return _parse_transcript(data)
