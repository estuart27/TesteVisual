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


@app.post("/gerar-tatuagem")
async def gerar_tatuagem(
    foto_corpo: UploadFile = File(...),
    foto_desenho: UploadFile = File(...),
    local_tattoo: str = Form(...)
):
    try:
        conteudo_corpo = await validar_imagem(foto_corpo)
        conteudo_desenho = await validar_imagem(foto_desenho)

        # Processamento (mesma lógica de compressão dos anteriores)
        imagem_corpo = Image.open(io.BytesIO(conteudo_corpo)).convert("RGB")
        imagem_corpo.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buffer_corpo = io.BytesIO()
        imagem_corpo.save(buffer_corpo, format="JPEG", quality=85)
        corpo_bytes = buffer_corpo.getvalue()

        imagem_desenho = Image.open(io.BytesIO(conteudo_desenho)).convert("RGBA")
        imagem_desenho.thumbnail((800, 800), Image.Resampling.LANCZOS)
        buffer_desenho = io.BytesIO()
        imagem_desenho.save(buffer_desenho, format="PNG") 
        desenho_bytes = buffer_desenho.getvalue()

        prompt_tattoo = f"""
        Você é um tatuador hiper-realista e especialista em simulação digital.
        Sua tarefa é tatuar o desenho da IMAGEM 2 na pele da pessoa da IMAGEM 1, especificamente na região: {local_tattoo}.

        INSTRUÇÕES ESTRITAS:
        1. A tatuagem deve seguir perfeitamente a anatomia, curvatura do músculo e perspectiva do membro ({local_tattoo}).
        2. Mescle a tinta com a textura da pele original. O desenho deve adotar os poros, a iluminação, o tom de pele e os pelinhos do local. NÃO deve parecer um adesivo colado por cima.
        3. Diminua levemente a opacidade do preto/cores para simular tinta cicatrizada sob a pele.
        4. NÃO altere o fundo da imagem, nem o formato do corpo da pessoa. Preserve 100% da imagem base.
        """
        
        resposta = client.models.generate_content(
            model="gemini-3.1-flash-image-preview",
            contents=[
                types.Part.from_text(text="[IMAGEM BASE - CORPO DO CLIENTE]"),
                types.Part.from_bytes(data=corpo_bytes, mime_type="image/jpeg"),
                types.Part.from_text(text="[IMAGEM DO DECALQUE - DESENHO DA TATUAGEM]"),
                types.Part.from_bytes(data=desenho_bytes, mime_type="image/png"),
                types.Part.from_text(text=prompt_tattoo),
            ],
            config=types.GenerateContentConfig(response_modalities=["IMAGE"], temperature=0.2)
        )

        for part in resposta.candidates[0].content.parts:
            if part.inline_data is not None:
                img_b64 = base64.b64encode(part.inline_data.data).decode("utf-8")
                return JSONResponse(content={"sucesso": True, "imagem_final": img_b64})

        return JSONResponse(content={"sucesso": False, "erro": "Falha ao gerar tatuagem."})
    except Exception as e:
        return JSONResponse(content={"sucesso": False, "erro": str(e)})


@app.get("/")
def ler_raiz():
    return {"mensagem": "API MotoSticker rodando!"}
