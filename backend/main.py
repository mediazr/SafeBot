"""
SafeBot — Agente conversacional de SafeData Ops
Electiva: Agentes Virtuales Inteligentes
Maestría en Análisis de Datos y Sistemas Inteligentes
Universidad Santo Tomás — Marlon Esteban Díaz Rojas — 2026

Stack: LangChain + Claude claude-sonnet-4-6 + FastAPI
Herramientas: API propia SafeData Ops (NUSE 123 + XGBoost v4 R²=0.954)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

from langchain_core.messages import SystemMessage

# ── Configuración ────────────────────────────────────────────────────
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
SAFEDATAOPS_API_URL = os.getenv("SAFEDATAOPS_API_URL",
                                "https://safedataops.onrender.com")

# ── Herramientas del agente ──────────────────────────────────────────

@tool
def consultar_riesgo_upz(anio: int = 2025, mes: int = 7) -> str:
    """
    Consulta la estimación de riesgo por UPZ en San Cristóbal usando el modelo
    XGBoost v4 (R²=0.954, 87.67% eficacia). Devuelve el ranking de riesgo
    compuesto (70% predicción de incidentes + 30% IVL lumínico) para todas
    las UPZs de la localidad. Usar cuando el usuario pregunte por riesgo,
    zonas peligrosas, predicciones o ranking de UPZs.
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

            resultado = f"Estimación de riesgo — San Cristóbal — {mes}/{anio}\n"
            resultado += f"Modelo: {data.get('modelo', 'XGBoost v4')}\n\n"
            resultado += "Ranking de UPZs por riesgo compuesto:\n"
            for i, u in enumerate(upzs, 1):
                nivel = u.get("nivel_riesgo", "")
                riesgo_pct = round(u.get("riesgo", 0) * 100, 1)
                pred = u.get("prediccion_incidentes", 0)
                estrato = u.get("estrato_promedio", "N/D")
                resultado += (
                    f"{i}. {u['upz']}: {riesgo_pct}% ({nivel})\n"
                    f"   Predicción: {pred:,.0f} incidentes | Estrato: {estrato}\n"
                )
            return resultado
    except Exception as e:
        return f"Error consultando API de riesgo: {e}"


@tool
def consultar_incidentes_nuse(anio: str = "2023",
                               localidad: str = None) -> str:
    """
    Consulta estadísticas reales del NUSE 123 (Número Único de Seguridad y
    Emergencias de Bogotá). Devuelve total de incidentes, distribución por
    localidad, tipos de incidente más frecuentes y tendencia mensual.
    Usar cuando el usuario pregunte por incidentes históricos, estadísticas
    del NUSE, tipos de delito o tendencias de seguridad en Bogotá.
    """
    try:
        params = {"anio": anio}
        if localidad:
            params["localidad"] = localidad.upper()
        with httpx.Client(timeout=30) as client:
            r = client.get(
                f"{SAFEDATAOPS_API_URL}/api/resumen",
                params=params
            )
            data = r.json()

        total = data.get("total_incidentes", 0)
        por_tipo = data.get("por_tipo", [])[:5]
        por_loc  = data.get("por_localidad", [])[:5]

        resultado  = f"Incidentes NUSE 123 — {anio}"
        if localidad:
            resultado += f" — {localidad}"
        resultado += f"\nTotal: {total:,} incidentes\n"
        resultado += f"Fuente: {data.get('fuente','NUSE 123')}\n\n"

        if por_tipo:
            resultado += "Top 5 tipos de incidente:\n"
            for t in por_tipo:
                resultado += f"  • {t['tipo']}: {t['incidentes']:,}\n"

        if por_loc and not localidad:
            resultado += "\nTop 5 localidades con más incidentes:\n"
            for l in por_loc:
                resultado += f"  • {l['localidad']}: {l['incidentes']:,}\n"

        return resultado
    except Exception as e:
        return f"Error consultando NUSE 123: {e}"


@tool
def consultar_ivl_luminarias() -> str:
    """
    Consulta el Índice de Vulnerabilidad Lumínica (IVL) por UPZ en
    San Cristóbal. El IVL mide la relación entre incidentes históricos y
    cobertura de alumbrado público (luminarias UAESP/IDECA). Mayor IVL
    indica mayor vulnerabilidad. Incluye el estrato socioeconómico promedio
    de cada UPZ según datos del DANE. Usar cuando el usuario pregunte
    por luminarias, alumbrado, vulnerabilidad o estratificación.
    """
    try:
        with httpx.Client(timeout=30) as client:
            r = client.get(f"{SAFEDATAOPS_API_URL}/api/ivl")
            data = r.json()

        upzs = data.get("upzs", [])
        resultado  = "Índice de Vulnerabilidad Lumínica (IVL) — San Cristóbal\n"
        resultado += f"Fórmula: {data.get('formula','IVL = Incidentes / (Luminarias + 1)')}\n"
        resultado += f"Fuente: {data.get('fuente','Luminarias UAESP/IDECA + DANE')}\n\n"

        upzs_sorted = sorted(upzs, key=lambda x: -x.get("VULNERABILITY_INDEX", 0))
        for u in upzs_sorted:
            ivl    = u.get("VULNERABILITY_INDEX", 0)
            lum    = u.get("INFRA_POINTS", 0)
            estrato= u.get("estrato_promedio", "N/D")
            resultado += (
                f"• {u['UPZ']}: IVL={ivl:,.0f} | "
                f"Luminarias={lum:.0f} | Estrato={estrato}\n"
            )
        return resultado
    except Exception as e:
        return f"Error consultando IVL: {e}"


@tool
def informacion_safedata_ops(tema: str = "general") -> str:
    """
    Proporciona información sobre SafeData Ops S.A.S.: qué es el sistema,
    cómo funciona el modelo estadístico, qué fuentes de datos usa, cuáles
    son sus productos y segmentos de clientes, y el contexto académico del
    proyecto. Usar cuando el usuario pregunte qué es SafeData Ops, cómo
    funciona, cuáles son sus productos o cuál es el propósito del sistema.
    """
    info = {
        "general": """
SafeData Ops S.A.S. es un sistema de inteligencia geoespacial para
estimación y visualización de riesgo urbano en Bogotá D.C.

PROPUESTA DE VALOR:
SafeData transforma datos históricos y geoespaciales heterogéneos en
estimaciones explicables de riesgo por zona, proporcionando información
para apoyar la priorización de recursos y la toma de decisiones de seguridad.

FUENTES DE DATOS (tres):
1. NUSE 123 — 1.510.324 incidentes · San Cristóbal · 2015-2026
   (Número Único de Seguridad y Emergencias · Datos Abiertos Bogotá)
2. Luminarias UAESP/IDECA — Alumbrado público · Spatial Join WGS84
3. Estratificación DANE — 4.202 manzanas · estrato por UPZ

MODELO ESTADÍSTICO:
• Algoritmo: XGBoost (Extreme Gradient Boosting)
• R² = 0.954 — explica el 95.4% de la variabilidad
• Eficacia de estimación: 87.67% (validación temporal estricta)
• MAE: 22.18 incidentes/UPZ/mes
• Features: 34 variables (temporalidad, lags históricos, IVL, estrato, tipo)

CICLO DEL SISTEMA:
01 Ingesta → 02 Integración geoespacial → 03 Estimación XGBoost → 04 Visualización explicable
""",
        "productos": """
PRODUCTOS DE SAFEDATA OPS:

1. SafeCity Command Center (B2G)
   • Clientes: Alcaldías locales · C4 Bogotá · Secretaría de Seguridad
   • Precio: $90M - $180M COP / año
   • Estado: En producción (prototipo)

2. SafeRoute Pro API (B2B Logística)
   • Clientes: Empresas de transporte · Seguridad privada
   • Precio: $6M COP / mes
   • Estado: En producción (prototipo)

3. SafeEstate Analytics (B2B Inmobiliario)
   • Clientes: Constructoras · Fondos de inversión
   • Precio: $3M COP / informe
   • Estado: En producción (prototipo)

4. SafeCitizen App (B2C) — PRODUCTO CONCEPTUAL
   • Clientes: Ciudadanos de Bogotá
   • Precio: Freemium
   • Estado: Concepto — versión futura (Etapa 3, año 5+)
""",
        "academico": """
CONTEXTO ACADÉMICO:

Proyecto de grado — Maestría en Análisis de Datos y Sistemas Inteligentes
Universidad Santo Tomás · Bucaramanga · 2026

Autor: Marlon Esteban Díaz Rojas
Directoras: Yuli Andrea Álvarez Pizarro · Pedro Pablo Díaz Jaimes

Modalidad: Opción de grado — Creación de empresa
Caso de estudio: Localidad San Cristóbal · Bogotá D.C.

Tecnologías: Python · XGBoost · FastAPI · React · GeoPandas
Despliegue: GitHub + Render (cloud)
Repositorio: github.com/mediazr/SafeDataOps
"""
    }
    return info.get(tema, info["general"])


# ── Agente LangChain ─────────────────────────────────────────────────
HERRAMIENTAS = [
    consultar_riesgo_upz,
    consultar_incidentes_nuse,
    consultar_ivl_luminarias,
    informacion_safedata_ops,
]

SYSTEM_PROMPT = """Eres SafeBot, el asistente virtual inteligente de SafeData Ops S.A.S.

SafeData Ops es un sistema de inteligencia geoespacial para estimación y
visualización de riesgo urbano en Bogotá D.C., desarrollado como proyecto
de grado de la Maestría en Análisis de Datos y Sistemas Inteligentes
de la Universidad Santo Tomás.

TU ROL:
- Responder preguntas sobre riesgo urbano en San Cristóbal usando datos reales
- Explicar cómo funciona el modelo estadístico XGBoost (R²=0.954)
- Orientar sobre los productos y servicios de SafeData Ops
- Proporcionar estadísticas reales del NUSE 123

HERRAMIENTAS DISPONIBLES:
- consultar_riesgo_upz: estimación de riesgo por UPZ para un mes/año
- consultar_incidentes_nuse: estadísticas históricas del NUSE 123
- consultar_ivl_luminarias: índice de vulnerabilidad lumínica por UPZ
- informacion_safedata_ops: información sobre el sistema y sus productos

INSTRUCCIONES:
- Responde SIEMPRE en español
- Usa las herramientas para dar datos reales, no inventes cifras
- Sé conciso pero informativo — máximo 3-4 párrafos por respuesta
- Cuando des datos de riesgo, explica qué significa para el usuario
- Si preguntan por zonas fuera de San Cristóbal, indica que el piloto
  actual cubre solo esa localidad pero el sistema es escalable
- Menciona siempre la fuente de los datos (NUSE 123, IDECA, DANE)

EJEMPLOS DE LO QUE PUEDES RESPONDER:
- Preguntas sobre riesgo por zona o UPZ
- Estadísticas de incidentes en Bogotá
- Información sobre el modelo estadístico y sus métricas
- Productos y precios de SafeData Ops
- Contexto académico del proyecto
- Cómo se integran las tres fuentes de datos

Empieza cada conversación presentándote brevemente."""

def crear_agente():
    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        api_key=ANTHROPIC_API_KEY,
        temperature=0.2,
        max_tokens=1000,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
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

# Cache de sesiones (memoria por conversación)
sesiones: dict = {}

def obtener_sesion(session_id: str):
    if session_id not in sesiones:
        sesiones[session_id] = {
            "agente": crear_agente(),
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
    mensaje: str
    session_id: str = "default"

class MensajeResponse(BaseModel):
    respuesta: str
    session_id: str

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "agente": "SafeBot v1.0",
        "llm": "Claude claude-sonnet-4-6 (Anthropic)",
        "framework": "LangChain",
        "herramientas": [t.name for t in HERRAMIENTAS],
        "safedata_api": SAFEDATAOPS_API_URL,
    }

@app.post("/api/chat", response_model=MensajeResponse)
async def chat(req: MensajeRequest):
    try:
        sesion   = obtener_sesion(req.session_id)
        agente   = sesion["agente"]
        historia = sesion["historia"]

        resultado = agente.invoke({
            "input": req.mensaje,
            "chat_history": historia,
        })

        respuesta = resultado.get("output", "Lo siento, no pude procesar tu pregunta.")

        # Actualizar historia
        from langchain_core.messages import HumanMessage, AIMessage
        historia.append(HumanMessage(content=req.mensaje))
        historia.append(AIMessage(content=respuesta))

        # Mantener solo las últimas 10 interacciones
        if len(historia) > 20:
            historia = historia[-20:]
            sesion["historia"] = historia

        return MensajeResponse(respuesta=respuesta, session_id=req.session_id)

    except Exception as e:
        return MensajeResponse(
            respuesta=f"Error procesando tu pregunta: {str(e)}. Por favor intenta de nuevo.",
            session_id=req.session_id
        )

@app.delete("/api/chat/{session_id}")
async def limpiar_sesion(session_id: str):
    if session_id in sesiones:
        del sesiones[session_id]
    return {"mensaje": "Sesión eliminada", "session_id": session_id}

# Servir frontend
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(os.path.join(frontend_path, "index.html")):
    @app.get("/")
    async def root():
        return FileResponse(os.path.join(frontend_path, "index.html"))

    @app.get("/{path:path}")
    async def static(path: str):
        fp = os.path.join(frontend_path, path)
        if os.path.exists(fp) and os.path.isfile(fp):
            return FileResponse(fp)
        return FileResponse(os.path.join(frontend_path, "index.html"))
