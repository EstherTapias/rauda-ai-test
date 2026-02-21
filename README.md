# 🎫 LLM-Based Ticket Reply Evaluator
**Rauda AI — Take-Home Assignment**

Evalúa respuestas de soporte al cliente usando un LLM, puntuando cada reply en
**contenido** (relevancia, exactitud, completitud) y **formato** (claridad, estructura,
gramática) en una escala del 1 al 5 con explicación textual.

> **Nota de implementación:** El assignment original especifica OpenAI GPT-4o.
> Esta solución usa **Llama 3.3 70B** vía **Groq API**. Cambiar a GPT-4o en producción
> requiere modificar únicamente el cliente y el nombre del modelo.

---

## 🏗️ Estructura del proyecto

```
rauda_ai_test/
├── data/
│   ├── tickets.csv
│   └── tickets_evaluated.csv
├── notebooks/
│   └── ticket_evaluator_groq.ipynb
├── src/
│   └── main.py
├── tests/
│   └── test_evaluator.py
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clona el repositorio

```bash
git clone <repo-url>
cd rauda_ai_test
```

### 2. Crea y activa el entorno virtual

```bash
python -m venv venv

# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Instala las dependencias

```bash
pip install -r requirements.txt
```

### 4. Registra el entorno en Jupyter para poder usarlo como kernel

```bash
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```

### 5. Consigue tu API Key gratuita de Groq

1. Ve a [console.groq.com](https://console.groq.com) y regístrate con Google o email
2. Ve a **"API Keys"** → **"Create API Key"**
3. No necesitas tarjeta de crédito

### 6. Configura las credenciales

Crea un archivo `.env` en la raíz del proyecto:

```
GROQ_API_KEY=gsk_...tu-clave-aqui...
```

> ⚠️ **Seguridad:** El archivo `.env` está en `.gitignore` y **nunca** debe subirse
> a GitHub. Si se expone públicamente, cualquiera puede usar tu clave y agotar tu cuota.
> Por eso usamos `python-dotenv` para cargar credenciales desde el entorno local,
> nunca hardcodeadas en el código.

---

## ▶️ Ejecución

### Opción A — Jupyter Notebook (recomendado)

```bash
jupyter notebook notebooks/ticket_evaluator_groq.ipynb
```

Selecciona el kernel **"Python (venv)"** y ejecuta las celdas en orden.
El resultado se guarda en `tickets_evaluated.csv`.

### Opción B — Script Python

```bash
python src/main.py
```

---

## 🧪 Tests

```bash
pytest tests/test_evaluator.py -v
```

Los tests cubren las funciones core **sin llamadas reales a la API** (usan `unittest.mock`):
- Lectura y validación del CSV de entrada
- Construcción del User Prompt con caracteres especiales y unicode
- Validación del schema de respuesta del LLM (campos, tipos, rango de scores)
- Escritura y estructura del CSV de salida

---

## 📦 Dependencias

| Librería | Uso |
|---|---|
| `groq` | Cliente oficial Groq API |
| `pandas` | Lectura y escritura de CSV |
| `python-dotenv` | Carga segura de variables de entorno desde `.env` |
| `tenacity` | Retries automáticos con backoff exponencial |
| `pytest` | Suite de tests unitarios |
| `jupyter` | Entorno de ejecución del notebook |

---

## 🏛️ Decisiones de arquitectura

**¿Por qué Groq + Llama 3.3 70B?**
Groq ofrece tier gratuito sin tarjeta, con API compatible con el estándar OpenAI.
Migrar a GPT-4o requiere cambiar únicamente el cliente y el nombre del modelo.

**¿Por qué JSON mode?**
`response_format={"type": "json_object"}` garantiza que el modelo devuelva siempre
JSON válido. Elimina la necesidad de regex frágiles para parsear texto libre.

**¿Por qué temperatura 0.1?**
Las evaluaciones deben ser reproducibles. Temperatura baja genera respuestas
consistentes — el mismo ticket evaluado dos veces debe dar scores similares.

**¿Por qué retries con backoff exponencial?**
`tenacity` reintenta ante errores 429/503 esperando 2s → 4s → 8s → 16s,
respetando los límites del proveedor sin saturarlo.

**¿Por qué fail-safe por fila?**
Si una fila falla permanentemente, se registra el error y el pipeline continúa.
Se procesan 499/500 tickets correctamente aunque uno falle.

---

## 🚀 Escalabilidad a 1 millón de tickets

### 1. Procesamiento concurrente con asyncio
La versión actual es secuencial. Con `asyncio` y `asyncio.Semaphore(50)` se
lanzarían 50 llamadas concurrentes, reduciendo el tiempo de días a horas.

### 2. Arquitectura de cola con AWS SQS + Lambda
- **S3** recibe el CSV → **Lambda** publica cada fila en **SQS**
- Múltiples **Lambda workers** procesan en paralelo → resultados en **DynamoDB**
- Ventajas: escalado automático, Dead Letter Queue para fallos, pay-per-execution

### 3. Control de costes y observabilidad
- Caché semántica (Redis) para tickets similares ya evaluados
- Modelo tiered: barato para el grueso, premium solo para scores 2-3 → ahorro ~80%
- Alertas de presupuesto + logging de tokens consumidos por llamada
