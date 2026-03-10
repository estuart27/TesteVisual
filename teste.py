import os
import io
import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = "AIzaSyDrgHpjyAE8-1VPRSUV6HIDq7QAFY55PQw"
client = genai.Client(api_key=GOOGLE_API_KEY)

ADESIVOS_CATALOGO = {
    "caveira_fogo": {
        "nome": "Caveira com chamas",
        "arquivo": "adesivos/caveira_fogo.png",
    },
    "raio_azul": {
        "nome": "Raio Azul",
        "arquivo": "adesivos/raio_azul.png",
    }
}

@app.post("/gerar-simulacao")
async def gerar_simulacao(
    foto_moto: UploadFile = File(...),
    id_adesivo: str = Form(...),
    local_adesivo: str = Form(...)   # ← novo campo vindo do front
):
    if id_adesivo not in ADESIVOS_CATALOGO:
        raise HTTPException(status_code=400, detail="Adesivo não encontrado.")

    info_adesivo = ADESIVOS_CATALOGO[id_adesivo]

    if not os.path.exists(info_adesivo["arquivo"]):
        raise HTTPException(status_code=500, detail=f"Arquivo do adesivo não encontrado: {info_adesivo['arquivo']}")

    try:
        # Prepara foto da moto
        conteudo_moto = await foto_moto.read()
        imagem_moto = Image.open(io.BytesIO(conteudo_moto)).convert("RGB")
        imagem_moto = imagem_moto.resize((1024, 1024))
        buffer_moto = io.BytesIO()
        imagem_moto.save(buffer_moto, format="PNG")
        moto_bytes = buffer_moto.getvalue()

        # Prepara imagem do adesivo
        imagem_adesivo = Image.open(info_adesivo["arquivo"]).convert("RGBA")
        buffer_adesivo = io.BytesIO()
        imagem_adesivo.save(buffer_adesivo, format="PNG")
        adesivo_bytes = buffer_adesivo.getvalue()

        prompt = f"""Você é um editor de imagens profissional especialista em customização de motos.

IMAGEM 1: foto da moto do cliente.
IMAGEM 2: o adesivo que ele deseja aplicar.

TAREFA: Cole o adesivo da IMAGEM 2 em: {local_adesivo} da moto da IMAGEM 1.

REGRAS OBRIGATÓRIAS:
- Use o adesivo EXATAMENTE como aparece na Imagem 2 — mesma arte, mesmas cores, mesmo formato
- NÃO recrie, NÃO substitua o adesivo por outro
- O adesivo deve seguir a curvatura e iluminação da superfície da moto
- Mantenha a moto, o fundo e todo o resto IDÊNTICO à Imagem 1
- Resultado fotorrealista, como se o adesivo fosse aplicado profissionalmente de fábrica"""

        resposta = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=moto_bytes, mime_type="image/png"),
                types.Part.from_bytes(data=adesivo_bytes, mime_type="image/png"),
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            )
        )

        for part in resposta.candidates[0].content.parts:
            if part.inline_data is not None:
                imagem_final_base64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                return JSONResponse(content={"sucesso": True, "imagem_final": imagem_final_base64})

        raise Exception("A IA não retornou nenhuma imagem.")

    except Exception as e:
        return JSONResponse(content={"sucesso": False, "erro": str(e)})

@app.get("/")
def ler_raiz():
    return {"mensagem": "API MotoSticker rodando!"}