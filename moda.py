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

MAX_FILE_SIZE = 5 * 1024 * 1024  
FORMATOS_PERMITIDOS = ["image/jpeg", "image/png", "image/webp"]

async def validar_imagem(file: UploadFile):
    if file.content_type not in FORMATOS_PERMITIDOS:
        raise HTTPException(status_code=400, detail=f"Formato não suportado. Use JPG, PNG ou WEBP.")
    conteudo = await file.read()
    if len(conteudo) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="A imagem excede o limite de 5MB.")
    return conteudo

@app.post("/gerar-provador")
async def gerar_provador(
    foto_pessoa: UploadFile = File(...),
    foto_roupa: UploadFile = File(...),
    tipo_peca: str = Form(...)
):
    try:
        conteudo_pessoa = await validar_imagem(foto_pessoa)
        conteudo_roupa = await validar_imagem(foto_roupa)

        # --- PROCESSAMENTO DA PESSOA ---
        try:
            imagem_pessoa = Image.open(io.BytesIO(conteudo_pessoa)).convert("RGB")
            imagem_pessoa.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            buffer_pessoa = io.BytesIO()
            imagem_pessoa.save(buffer_pessoa, format="JPEG", quality=85)
            pessoa_bytes = buffer_pessoa.getvalue()
        except UnidentifiedImageError:
            return JSONResponse(content={"sucesso": False, "erro": "Arquivo da pessoa corrompido ou inválido."})

        # --- PROCESSAMENTO DA ROUPA ---
        try:
            imagem_roupa = Image.open(io.BytesIO(conteudo_roupa)).convert("RGBA")
            imagem_roupa.thumbnail((800, 800), Image.Resampling.LANCZOS)
            buffer_roupa = io.BytesIO()
            imagem_roupa.save(buffer_roupa, format="PNG") 
            roupa_bytes = buffer_roupa.getvalue()
        except UnidentifiedImageError:
            return JSONResponse(content={"sucesso": False, "erro": "Arquivo da roupa corrompido ou inválido."})

        # --- PROMPT VIRTUAL TRY-ON ---
        prompt = f"""
        Você é um estilista virtual de alta costura e especialista em edição fotorealista (Virtual Try-On).
        Sua tarefa é vestir a roupa da IMAGEM 2 na pessoa da IMAGEM 1, atuando como: {tipo_peca}.

        INSTRUÇÕES ESTRITAS (CRÍTICO):
        1. PRESERVE A IDENTIDADE: O rosto, cabelo, tom de pele e características corporais da pessoa na IMAGEM 1 NÃO PODEM SER ALTERADOS. O fundo também deve continuar 100% idêntico.
        2. A roupa da IMAGEM 2 deve ser ajustada ao corpo da pessoa de forma realista, respeitando a pose, perspectiva, volumes corporais e a iluminação do ambiente original.
        3. Preserve o design, cor, textura e estampas originais da peça de roupa.
        4. Crie dobras (drapeado) e sombras naturais no tecido para que não pareça um adesivo plano, mas sim uma roupa vestida no corpo.
        """
        
        resposta = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=[
                types.Part.from_text(text="[IMAGEM DE REFERÊNCIA - PESSOA (CLIENTE)]"),
                types.Part.from_bytes(data=pessoa_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text="[IMAGEM DA ROUPA - MANTER DESIGN E CORES EXATOS]"),
                types.Part.from_bytes(data=roupa_bytes, mime_type="image/png"),
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

        return JSONResponse(content={"sucesso": False, "erro": "A IA não conseguiu gerar a montagem da roupa."})

    except Exception as e:
        print(f"Erro interno: {str(e)}") 
        return JSONResponse(content={"sucesso": False, "erro": "Não foi possível gerar o provador no momento. Tente novamente."})