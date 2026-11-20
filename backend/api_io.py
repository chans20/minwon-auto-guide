"""core.py 를 HTTP로 쓰고 싶을 때 (화면이 웹/JS 이거나 서버가 따로일 때)

실행:  uvicorn api:app --host 0.0.0.0 --port 8000
확인:  http://<서버IP>:8000/docs  (Swagger 화면에서 바로 테스트 가능)
"""
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

import core

app = FastAPI(title='민원 입출력 모듈')

# 앱(웹/모바일)이 다른 주소에서 호출할 수 있게 허용 (교육용이라 전체 허용)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])


class TTSRequest(BaseModel):
    text: str
    speaker: str = 'nara'
    speed: int = 0


@app.post('/input')
async def input_(text: Optional[str] = Form(None), image: Optional[UploadFile] = File(None)):
    """텍스트 또는 이미지 → {"text": ..., "source": "text"|"ocr"}"""
    try:
        if image is not None:
            return core.normalize_input(text, await image.read(), image.filename)
        return core.normalize_input(text)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(502, str(e))


@app.post('/tts', response_class=Response)
def tts(req: TTSRequest):
    """행동강령 텍스트 → mp3"""
    try:
        audio = core.synthesize_speech(req.text, req.speaker, req.speed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    return Response(audio, media_type='audio/mpeg')
