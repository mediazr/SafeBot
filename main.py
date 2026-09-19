"""
SafeBot - Agente conversacional de SafeData Ops
Electiva: Agentes Virtuales Inteligentes
Maestria en Analisis de Datos y Sistemas Inteligentes
Universidad Santo Tomas - Marlon Esteban Diaz Rojas - 2026

Stack: LangChain + Claude claude-sonnet-4-6 + FastAPI
Herramientas: API propia SafeData Ops (NUSE 123 + XGBoost v4 R2=0.954)
RAG: Libro de tesis completo como contexto del agente
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import os
import json
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ── Configuracion ────────────────────────────────────────────────────
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
SAFEDATAOPS_API_URL = os.getenv("SAFEDATAOPS_API_URL",
                                "https://safedataops.onrender.com")

# ── BASE_DIR siempre primero ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Cargar libro de tesis (RAG) ──────────────────────────────────────
_THESIS_TEXT = ''
try:
    _thesis_path = os.path.join(BASE_DIR, 'thesis_context.txt')
    with open(_thesis_path, encoding='utf-8') as _f:
        _THESIS_TEXT = _f.read()
    print(f"Libro de tesis cargado: {len(_THESIS_TEXT.split()):,} palabras")
except Exception as _e:
    print(f"Libro no disponible: {_e}")

# ── Herramientas del agente ──────────────────────────────────────────
@tool
def consultar_riesgo_upz(anio: int = 2025, mes: int = 7) -> str:
    """
    Consulta la estimacion de riesgo por UPZ en San Cristobal usando el modelo
    XGBoost v4 (R2=0.954, 87.67% eficacia). Devuelve el ranking de riesgo
    compuesto (70% prediccion de incidentes + 30% IVL luminico) para todas
    las UPZs. Usar cuando pregunten por riesgo, zonas peligrosas o predicciones.
    """
    try:
        with httpx.Client(timeout=30) as client:
            r = client.get(
                f"{SAFEDATAOPS_API_URL}/api/riesgo/san-cristobal",
                params={"anio": anio, "mes": mes}
            )
            data = r.json()
            upzs = data.get("upzs", [])
            if not upzs:
                return "No se obtuvieron datos de riesgo."

            resultado = f"Estimacion de riesgo - San Cristobal - {mes}/{anio}\n"
            resultado += f"Modelo: {data.get('modelo', 'XGBoost v4')}\n\n"
            resultado += "Ranking de UPZs por riesgo compuesto:\n"
            for i, u in enumerate(upzs, 1):
                nivel    = u.get("nivel_riesgo", "")
                riesgo   = round(u.get("riesgo", 0) * 100, 1)
                pred     = u.get("prediccion_incidentes", 0)
                estrato  = u.get("estrato_promedio", "N/D")
                resultado += (
                    f"{i}. {u['upz']}: {riesgo}% ({nivel})\n"
                    f"   Prediccion: {pred:,.0f} incidentes | Estrato: {estrato}\n"
                )
            return resultado
    except Exception as e:
        return f"Error consultando API de riesgo: {e}"


@tool
def consultar_incidentes_nuse(anio: str = "2023", localidad: str = None) -> str:
    """
    Consulta estadisticas reales del NUSE 123. Devuelve total de incidentes,
    distribucion por localidad y tipos de incidente. Usar cuando pregunten
    por incidentes historicos, estadisticas o tendencias de seguridad.
    """
    try:
        params = {"anio": anio}
        if localidad:
            params["localidad"] = localidad.upper()
        with httpx.Client(timeout=30) as client:
            r    = client.get(f"{SAFEDATAOPS_API_URL}/api/resumen", params=params)
            data = r.json()

        total    = data.get("total_incidentes", 0)
        por_tipo = data.get("por_tipo", [])[:5]
        por_loc  = data.get("por_localidad", [])[:5]

        resultado  = f"Incidentes NUSE 123 - {anio}"
        if localidad:
            resultado += f" - {localidad}"
        resultado += f"\nTotal: {total:,} incidentes\n"
        resultado += f"Fuente: {data.get('fuente', 'NUSE 123')}\n\n"

        if por_tipo:
            resultado += "Top 5 tipos de incidente:\n"
            for t in por_tipo:
                resultado += f"  * {t['tipo']}: {t['incidentes']:,}\n"

        if por_loc and not localidad:
            resultado += "\nTop 5 localidades:\n"
            for l in por_loc:
                resultado += f"  * {l['localidad']}: {l['incidentes']:,}\n"

        return resultado
    except Exception as e:
        return f"Error consultando NUSE 123: {e}"


@tool
def consultar_ivl_luminarias() -> str:
    """
    Consulta el Indice de Vulnerabilidad Luminica (IVL) por UPZ en San Cristobal.
    Incluye estrato socioeconomico DANE. Usar cuando pregunten por luminarias,
    alumbrado, vulnerabilidad o estratificacion socioeconomica.
    """
    try:
        with httpx.Client(timeout=30) as client:
            r    = client.get(f"{SAFEDATAOPS_API_URL}/api/ivl")
            data = r.json()

        upzs = data.get("upzs", [])
        resultado  = "Indice de Vulnerabilidad Luminica (IVL) - San Cristobal\n"
        resultado += "Formula: IVL = Incidentes / (Luminarias funcionales + 1)\n"
        resultado += "Fuente: Luminarias UAESP/IDECA + Estratificacion DANE\n\n"

        upzs_sorted = sorted(upzs, key=lambda x: -x.get("VULNERABILITY_INDEX", 0))
        for u in upzs_sorted:
            ivl     = u.get("VULNERABILITY_INDEX", 0)
            lum     = u.get("INFRA_POINTS", 0)
            estrato = u.get("estrato_promedio", "N/D")
            resultado += (
                f"* {u['UPZ']}: IVL={ivl:,.0f} | "
                f"Luminarias={lum:.0f} | Estrato={estrato}\n"
            )
        return resultado
    except Exception as e:
        return f"Error consultando IVL: {e}"


@tool
def informacion_safedata_ops(tema: str = "general") -> str:
    """
    Proporciona informacion sobre SafeData Ops: que es, como funciona,
    productos, segmentos de clientes y contexto academico. Usar cuando
    pregunten por la empresa, sus productos o el proyecto de grado.
    """
    info = {
        "general": """
SafeData Ops S.A.S. es un sistema de inteligencia geoespacial para
estimacion y visualizacion de riesgo urbano en Bogota D.C.

PROPUESTA DE VALOR:
SafeData transforma datos historicos y geoespaciales heterogeneos en
estimaciones explicables de riesgo por zona, proporcionando informacion
para apoyar la priorizacion de recursos y la toma de decisiones de seguridad.

TRES FUENTES DE DATOS:
1. NUSE 123 - 1.510.324 incidentes San Cristobal 2015-2026
2. Luminarias UAESP/IDECA - Alumbrado publico - Spatial Join WGS84
3. Estratificacion DANE - 4.202 manzanas - estrato por UPZ

MODELO ESTADISTICO:
* Algoritmo: XGBoost (Extreme Gradient Boosting)
* R2 = 0.954 (explica el 95.4% de la variabilidad)
* Eficacia de estimacion: 87.67% (validacion temporal estricta)
* MAE: 22.18 incidentes/UPZ/mes
* Features: 34 variables

CICLO DEL SISTEMA:
01 Ingesta -> 02 Integracion geoespacial -> 03 Estimacion XGBoost -> 04 Visualizacion
""",
        "productos": """
PRODUCTOS DE SAFEDATA OPS:

1. SafeCity Command Center (B2G)
   * Clientes: Alcaldias locales, C4 Bogota, Secretaria de Seguridad
   * Precio: $90M - $180M COP / ano
   * Estado: En produccion (prototipo)

2. SafeRoute Pro API (B2B Logistica)
   * Clientes: Empresas de transporte, Seguridad privada
   * Precio: $6M COP / mes
   * Estado: En produccion (prototipo)

3. SafeEstate Analytics (B2B Inmobiliario)
   * Clientes: Constructoras, Fondos de inversion
   * Precio: $3M COP / informe
   * Estado: En produccion (prototipo)

4. SafeCitizen App (B2C) - PRODUCTO CONCEPTUAL
   * Clientes: Ciudadanos de Bogota
   * Precio: Freemium
   * Estado: Concepto - version futura (Etapa 3, ano 5+)
""",
        "financiero": """
MODELO FINANCIERO SAFEDATA OPS:

* Inversion inicial: $210M COP
  (60% credito bancario 17.55% EA + 40% capital semilla)
* VPN: $1.112 billones COP
* TIR: 78.1%
* TIRM: 58.9%
* B/C: 6.3x
* Periodo de recuperacion: 2.5 anos (escenario realista)
* Tres escenarios: optimista, realista y conservador - todos VIABLES

PROYECCION DE INGRESOS:
* Ano 1: $246M COP
* Ano 5: $2.160M COP
""",
        "academico": """
CONTEXTO ACADEMICO:

Proyecto de grado - Maestria en Analisis de Datos y Sistemas Inteligentes
Universidad Santo Tomas - Bucaramanga - 2026
Modalidad: Opcion de grado - Creacion de empresa

Autor: Marlon Esteban Diaz Rojas
Directoras: Yuli Andrea Alvarez Pizarro y Pedro Pablo Diaz Jaimes
Caso de estudio: Localidad San Cristobal - Bogota D.C.

Tecnologias: Python, XGBoost, FastAPI, React, GeoPandas, LangChain
Despliegue: GitHub + Render
Repositorio: github.com/mediazr/SafeDataOps
"""
    }
    return info.get(tema, info["general"])


# ── System Prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres SafeBot, el asistente virtual inteligente de SafeData Ops S.A.S.,
un sistema de inteligencia geoespacial para estimacion y visualizacion de riesgo urbano
en Bogota D.C., desarrollado como proyecto de grado de la Maestria en Analisis de Datos
y Sistemas Inteligentes de la Universidad Santo Tomas por Marlon Esteban Diaz Rojas.

TU ROL:
- Responder preguntas sobre riesgo urbano en San Cristobal usando datos reales
- Explicar como funciona el modelo estadistico XGBoost (R2=0.954)
- Orientar sobre los productos y servicios de SafeData Ops
- Responder sobre el proyecto academico, competidores y modelo de negocio
- Usar el conocimiento base del plan de negocio cuando sea relevante

HERRAMIENTAS DISPONIBLES:
- consultar_riesgo_upz: estimacion de riesgo por UPZ para un mes/ano
- consultar_incidentes_nuse: estadisticas historicas del NUSE 123
- consultar_ivl_luminarias: indice de vulnerabilidad luminica + estrato DANE
- informacion_safedata_ops: informacion sobre la empresa y sus productos

INSTRUCCIONES:
- Responde SIEMPRE en espanol
- Usa las herramientas para dar datos reales, no inventes cifras
- Se conciso pero informativo - maximo 3-4 parrafos por respuesta
- Menciona la fuente de los datos (NUSE 123, IDECA, DANE)
- Si preguntan por zonas fuera de San Cristobal, indica que el piloto
  actual cubre solo esa localidad pero el sistema es escalable a toda Bogota"""


def get_system_prompt():
    """Retorna el system prompt con el libro de tesis como contexto RAG."""
    if _THESIS_TEXT:
        return (
            SYSTEM_PROMPT
            + "\n\nCONOCIMIENTO BASE - PLAN DE NEGOCIO SAFEDATA OPS:\n"
            + "El siguiente texto contiene el plan de negocio completo, "
            + "analisis de competidores, modelo financiero, descripcion tecnica "
            + "y conclusiones. Usalo para responder sobre el proyecto, "
            + "empresa, competidores y modelo de negocio.\n\n"
            + _THESIS_TEXT
        )
    return SYSTEM_PROMPT


# ── Agente LangChain ─────────────────────────────────────────────────
HERRAMIENTAS = [
    consultar_riesgo_upz,
    consultar_incidentes_nuse,
    consultar_ivl_luminarias,
    informacion_safedata_ops,
]


def crear_agente():
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=ANTHROPIC_API_KEY,
        temperature=0.2,
        max_tokens=1000,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", get_system_prompt()),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, HERRAMIENTAS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=HERRAMIENTAS,
        verbose=True,
        max_iterations=4,
        handle_parsing_errors=True,
    )


# Cache de sesiones
sesiones: dict = {}


def obtener_sesion(session_id: str):
    if session_id not in sesiones:
        sesiones[session_id] = {
            "agente":   crear_agente(),
            "historia": []
        }
    return sesiones[session_id]


# ── FastAPI ──────────────────────────────────────────────────────────
app = FastAPI(title="SafeBot API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class MensajeRequest(BaseModel):
    mensaje:    str
    session_id: str = "default"


class MensajeResponse(BaseModel):
    respuesta:  str
    session_id: str


@app.get("/api/health")
async def health():
    return {
        "status":        "ok",
        "agente":        "SafeBot v1.0",
        "llm":           "Claude claude-sonnet-4-6 (Anthropic)",
        "framework":     "LangChain",
        "herramientas":  [t.name for t in HERRAMIENTAS],
        "tesis_cargada": bool(_THESIS_TEXT),
        "safedata_api":  SAFEDATAOPS_API_URL,
    }


@app.post("/api/chat", response_model=MensajeResponse)
async def chat(req: MensajeRequest):
    try:
        sesion   = obtener_sesion(req.session_id)
        agente   = sesion["agente"]
        historia = sesion["historia"]

        resultado = agente.invoke({
            "input":        req.mensaje,
            "chat_history": historia,
        })

        # Extraer texto de la respuesta (puede ser string o lista de bloques)
        raw = resultado.get("output", "")
        if isinstance(raw, list):
            partes = []
            for bloque in raw:
                if isinstance(bloque, dict):
                    partes.append(bloque.get("text", ""))
                elif isinstance(bloque, str):
                    partes.append(bloque)
            respuesta = "".join(partes).strip()
        elif isinstance(raw, str):
            respuesta = raw.strip()
        else:
            respuesta = str(raw)

        if not respuesta:
            respuesta = "Lo siento, no pude procesar tu pregunta. Intenta de nuevo."

        # Actualizar historia (max 20 mensajes)
        historia.append(HumanMessage(content=req.mensaje))
        historia.append(AIMessage(content=respuesta))
        if len(historia) > 20:
            sesion["historia"] = historia[-20:]

        return MensajeResponse(respuesta=respuesta, session_id=req.session_id)

    except Exception as e:
        return MensajeResponse(
            respuesta=f"Error procesando tu pregunta: {str(e)}. Intenta de nuevo.",
            session_id=req.session_id
        )


@app.delete("/api/chat/{session_id}")
async def limpiar_sesion(session_id: str):
    if session_id in sesiones:
        del sesiones[session_id]
    return {"mensaje": "Sesion eliminada", "session_id": session_id}


# ── Servir frontend ──────────────────────────────────────────────────
@app.get("/")
async def root():
    idx = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"status": "ok", "message": "SafeBot API activa"}


@app.get("/{path:path}")
async def static(path: str):
    if path.startswith("api"):
        raise HTTPException(404, "Not found")
    fp = os.path.join(BASE_DIR, path)
    if os.path.exists(fp) and os.path.isfile(fp):
        return FileResponse(fp)
    idx = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    raise HTTPException(404, "Not found")
