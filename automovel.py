import os
import io
import base64
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("API_GEMINI")
if not GOOGLE_API_KEY:
    raise RuntimeError("A variável de ambiente GOOGLE_API_KEY não está definida.")

client = genai.Client(api_key=GOOGLE_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constantes de Validação
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 Megabytes
FORMATOS_PERMITIDOS = ["image/jpeg", "image/png", "image/webp"]

async def validar_imagem(file: UploadFile):
    # 1. Valida formato
    if file.content_type not in FORMATOS_PERMITIDOS:
        raise HTTPException(status_code=400, detail=f"Formato não suportado: {file.content_type}. Use JPG, PNG ou WEBP.")
    
    # 2. Lendo o arquivo e validando o tamanho
    conteudo = await file.read()
    if len(conteudo) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="A imagem excede o limite de 5MB.")
    
    return conteudo


# ==========================================
# ROTA 5: TUNING E ENVELOPAMENTO AUTOMOTIVO
# ==========================================
@app.post("/gerar-tuning")
async def gerar_tuning(
    foto_carro: UploadFile = File(...),
    foto_acessorio: UploadFile = File(...),
    tipo_modificacao: str = Form(...)
):
    try:
        conteudo_carro = await validar_imagem(foto_carro)
        conteudo_acessorio = await validar_imagem(foto_acessorio)

        imagem_carro = Image.open(io.BytesIO(conteudo_carro)).convert("RGB")
        imagem_carro.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
        buffer_carro = io.BytesIO()
        imagem_carro.save(buffer_carro, format="JPEG", quality=85)
        carro_bytes = buffer_carro.getvalue()

        imagem_acessorio = Image.open(io.BytesIO(conteudo_acessorio)).convert("RGBA")
        imagem_acessorio.thumbnail((800, 800), Image.Resampling.LANCZOS)
        buffer_acessorio = io.BytesIO()
        imagem_acessorio.save(buffer_acessorio, format="PNG") 
        acessorio_bytes = buffer_acessorio.getvalue()

        prompt_tuning = f"""
        Você é um especialista em renderização automotiva 3D e customização de carros (Tuning).
        Sua tarefa é aplicar a modificação/acessório da IMAGEM 2 no carro da IMAGEM 1, focado na região/tipo: {tipo_modificacao}.

        INSTRUÇÕES ESTRITAS E PROFISSIONAIS:
        1. PRESERVE O CARRO: O modelo do carro, a cor original (onde não houver modificação), o fundo e o cenário devem permanecer 100% idênticos. NÃO mude o formato do veículo.
        2. SE FOR RODA: Substitua as rodas originais pela IMAGEM 2. Respeite perfeitamente a perspectiva do ângulo do carro, o encaixe dentro da caixa de roda, e aplique sombras realistas nos pneus e reflexos metálicos no aro.
        3. SE FOR ENVELOPAMENTO/ADESIVO: Aplique o design seguindo a curvatura da lataria, dobras e vincos do carro. É OBRIGATÓRIO preservar os reflexos de luz e sombra do ambiente por cima do adesivo para parecer pintura/vinil automotivo real.
        4. O resultado deve parecer uma foto tirada na rua após a customização, não uma montagem amadora.
        """
        
        resposta = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=[
                types.Part.from_text(text="[IMAGEM BASE - CARRO DO CLIENTE]"),
                types.Part.from_bytes(data=carro_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text="[IMAGEM DO ACESSÓRIO / RODA / ARTE]"),
                types.Part.from_bytes(data=acessorio_bytes, mime_type="image/png"),
                types.Part.from_text(text=prompt_tuning),
            ],
            config=types.GenerateContentConfig(response_modalities=["IMAGE"], temperature=0.15)
        )

        for part in resposta.candidates[0].content.parts:
            if part.inline_data is not None:
                img_b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                return JSONResponse(content={"sucesso": True, "imagem_final": img_b64})

        return JSONResponse(content={"sucesso": False, "erro": "Falha ao gerar o projeto do carro."})
    except Exception as e:
        return JSONResponse(content={"sucesso": False, "erro": str(e)})

@app.get("/")
def ler_raiz():
    return {"mensagem": "API MotoSticker rodando!"}
