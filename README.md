# SafeBot — Agente Virtual Inteligente de SafeData Ops

**Electiva: Agentes Virtuales Inteligentes**  
Maestría en Análisis de Datos y Sistemas Inteligentes  
Universidad Santo Tomás · Marlon Esteban Díaz Rojas · 2026

---

## ¿Qué es SafeBot?

SafeBot es un agente conversacional inteligente que permite consultar en lenguaje
natural información sobre riesgo urbano en Bogotá D.C., conectándose a la API
de SafeData Ops para devolver datos reales.

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| **LLM** | Claude claude-sonnet-4-6 (Anthropic) |
| **Framework de agente** | LangChain (Tool Calling Agent) |
| **Backend** | Python + FastAPI |
| **Frontend** | HTML + CSS + JavaScript vanilla |
| **Despliegue** | Render |
| **API de datos** | SafeData Ops (NUSE 123 + XGBoost v4) |

## Herramientas del agente

| Herramienta | Descripción |
|---|---|
| `consultar_riesgo_upz` | Estimación de riesgo por UPZ vía modelo XGBoost v4 |
| `consultar_incidentes_nuse` | Estadísticas del NUSE 123 por año/localidad |
| `consultar_ivl_luminarias` | Índice de Vulnerabilidad Lumínica + estrato DANE |
| `informacion_safedata_ops` | Contexto del sistema, productos y proyecto académico |

## Fuentes de datos

- **NUSE 123** — 1.510.324 incidentes · San Cristóbal · 2015-2026
- **Luminarias UAESP/IDECA** — Alumbrado público · Spatial Join WGS84
- **Estratificación DANE** — 4.202 manzanas · estrato por UPZ

## Despliegue local

```bash
cd backend
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
export SAFEDATAOPS_API_URL=https://safedataops.onrender.com
uvicorn main:app --reload --port 8000
# Abrir frontend/index.html en el navegador
```

## Despliegue en Render

1. Subir repositorio a GitHub
2. Render → New Web Service → conectar repo
3. Agregar variable de entorno: `ANTHROPIC_API_KEY`
4. Deploy

## Conexión con el proyecto de grado

SafeBot es la capa de interacción conversacional del sistema SafeData Ops S.A.S.,
proyecto de grado de la Maestría en Análisis de Datos y Sistemas Inteligentes.
El agente consume la misma API que el dashboard web, demostrando que los datos
y el modelo estadístico son accesibles desde múltiples canales de interacción.

**Repositorio principal:** github.com/mediazr/SafeDataOps
