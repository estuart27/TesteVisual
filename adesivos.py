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

@app.post("/gerar-simulacao")
async def gerar_simulacao(
    foto_moto: UploadFile = File(...),
    foto_adesivo: UploadFile = File(...),
    local_adesivo: str = Form(...)
):
    try:
        # --- VALIDAÇÕES DE ENTRADA ---
        conteudo_moto = await validar_imagem(foto_moto)
        conteudo_adesivo = await validar_imagem(foto_adesivo)

        # --- PROCESSAMENTO DA MOTO ---
        try:
            imagem_moto = Image.open(io.BytesIO(conteudo_moto)).convert("RGB")
            imagem_moto.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            buffer_moto = io.BytesIO()
            imagem_moto.save(buffer_moto, format="JPEG", quality=85) # JPEG reduz o peso final que vai pra API
            moto_bytes = buffer_moto.getvalue()
        except UnidentifiedImageError:
            return JSONResponse(content={"sucesso": False, "erro": "Arquivo da moto corrompido ou inválido."})

        # --- PROCESSAMENTO DO ADESIVO ---
        try:
            imagem_adesivo = Image.open(io.BytesIO(conteudo_adesivo)).convert("RGBA")
            imagem_adesivo.thumbnail((800, 800), Image.Resampling.LANCZOS) # Adesivo não precisa ser gigante
            buffer_adesivo = io.BytesIO()
            imagem_adesivo.save(buffer_adesivo, format="PNG") # PNG para manter a transparência!
            adesivo_bytes = buffer_adesivo.getvalue()
        except UnidentifiedImageError:
            return JSONResponse(content={"sucesso": False, "erro": "Arquivo do adesivo corrompido ou inválido."})

        # --- PROMPT E INTEGRAÇÃO GEMINI ---
        prompt = f"""
        Você é um sistema automatizado de renderização fotorealista de veículos.
        Sua tarefa é compor a IMAGEM 2 (adesivo) sobre a IMAGEM 1 (moto), especificamente na região: {local_adesivo}.

        INSTRUÇÕES ESTRITAS:
        1. NÃO altere o modelo, cor, fundo ou qualquer detalhe da moto original. A moto deve permanecer 100% idêntica.
        2. O adesivo deve ser projetado na superfície ({local_adesivo}), respeitando a perspectiva, curvatura e iluminação da peça original.
        3. Preserve EXATAMENTE o design, textos e proporções do adesivo. Não invente ou distorça.
        4. Faça uma mesclagem suave das bordas do adesivo para parecer aplicação real.
        """
        
        resposta = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=[
                types.Part.from_text(text="[IMAGEM DE REFERÊNCIA - MOTO DO CLIENTE]"),
                types.Part.from_bytes(data=moto_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text="[IMAGEM DO ADESIVO - MANTER DESIGN E CORES EXATOS]"),
                types.Part.from_bytes(data=adesivo_bytes, mime_type="image/png"),
                types.Part.from_text(text=prompt),
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                temperature=0.2 
            )
        )

        for part in resposta.candidates[0].content.parts:
            if part.inline_data is not None:
                imagem_final_base64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                return JSONResponse(content={"sucesso": True, "imagem_final": imagem_final_base64})

        return JSONResponse(content={"sucesso": False, "erro": "A IA gerou a resposta, mas não incluiu a imagem final."})

    except Exception as e:
        # Tratamento de erro geral (Timeout da API, erro de chave, etc)
        print(f"Erro interno: {str(e)}") # Loga no terminal do servidor
        return JSONResponse(content={"sucesso": False, "erro": "Não foi possível gerar a simulação no momento. Tente novamente."})

@app.get("/")
def ler_raiz():
    return {"mensagem": "API MotoSticker rodando!"}