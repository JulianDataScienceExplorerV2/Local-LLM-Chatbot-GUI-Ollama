<div align="center">

# ✦ Ollama Chat Interface

**A sleek, local AI chat app built with Python & CustomTkinter**  
**Una app de chat IA local, hecha con Python y CustomTkinter**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://python.org)
[![CustomTkinter](https://img.shields.io/badge/CustomTkinter-5.2.2-purple?style=flat-square)](https://github.com/TomSchimansky/CustomTkinter)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.74-cyan?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green?style=flat-square)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

</div>

---

## 🖼️ Screenshots

> Chat view with multi-session sidebar / Vista de chat con sidebar de sesiones múltiples

<img width="1258" height="807" alt="image" src="https://github.com/user-attachments/assets/83f710dd-3957-4853-89e4-a58e26c75a37" />



---

## 🇬🇧 English

### What is this?

Ollama Chat Interface is a **local, private desktop app** that lets you chat with any AI model running on [Ollama](https://ollama.com) — think of it like a personal ChatGPT that runs entirely on your machine, no internet, no subscriptions, no data leaving your computer.

It's built with Python using **CustomTkinter** for the modern UI, **LangGraph** for conversation memory management, and the **Ollama** API for model inference.

### ✨ Features

- 🗂️ **Multi-session chat** — create, switch, and delete independent conversations
- 🤖 **Model selector** — pick any model installed in Ollama from the sidebar dropdown
- 💬 **Bubble-style UI** — user messages on the right, AI on the left, just like a real messenger
- ✦ **Typing indicator** — animated neon dots while the model is thinking
- 📝 **Basic Markdown rendering** — supports `**bold**`, `` `inline code` ``, and ` ```code blocks``` `
- 💾 **Export chat** — save any conversation as `.md` or `.txt`
- 🗑️ **Clear session** — wipe the history of any chat without losing others
- 🧠 **Persistent context** — uses LangGraph's `MemorySaver` to keep conversation history per session
- 🎨 **Synthwave dark theme** — purple, cyan & neon pink palette built with CustomTkinter

### 🚀 Getting Started

**Requirements:**
- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

```bash
# 1. Clone the repo
git clone https://github.com/JulianDataScienceExplorerV2/Chat-Interface-GUI-Ollama-Py.git
cd Chat-Interface-GUI-Ollama-Py

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama (in a separate terminal)
ollama serve

# 4. Pull a model if you don't have one yet
ollama pull llama3.2

# 5. Run the app
python main.py
```

### 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `Python 3.10+` | Core language |
| `CustomTkinter 5.2.2` | Modern UI widgets |
| `LangGraph 0.2.74` | Conversation graph & memory |
| `LangChain Ollama` | Ollama model integration |
| `Requests` | Fetch available models from Ollama API |

---

## 🇪🇸 Español

### ¿Qué es esto?

Ollama Chat Interface es una **app de escritorio local y privada** para chatear con cualquier modelo de IA que tengas en [Ollama](https://ollama.com). Es básicamente un ChatGPT personal que corre completamente en tu máquina — sin internet, sin suscripciones, sin que tus datos salgan a ningún lado.

Está hecha en Python con **CustomTkinter** para la interfaz moderna, **LangGraph** para manejar la memoria de la conversación, y la **API de Ollama** para la inferencia de modelos.

### ✨ Funcionalidades

- 🗂️ **Multi-sesión** — crea, cambia y elimina conversaciones independientes
- 🤖 **Selector de modelo** — elige cualquier modelo instalado en Ollama desde el sidebar
- 💬 **UI tipo messenger** — mensajes del usuario a la derecha, respuestas de la IA a la izquierda
- ✦ **Indicador de escritura** — puntos neón animados mientras el modelo piensa
- 📝 **Markdown básico** — soporta `**negrita**`, `` `código inline` `` y ` ```bloques de código``` `
- 💾 **Exportar chat** — guarda cualquier conversación como `.md` o `.txt`
- 🗑️ **Borrar sesión** — limpia el historial de un chat sin afectar los demás
- 🧠 **Contexto persistente** — usa `MemorySaver` de LangGraph para mantener el historial por sesión
- 🎨 **Tema dark synthwave** — paleta de púrpura, cyan y rosa neón con CustomTkinter

### 🚀 Cómo usarlo

**Requisitos:**
- Python 3.10+
- [Ollama](https://ollama.com) instalado y corriendo localmente

```bash
# 1. Clona el repo
git clone https://github.com/JulianDataScienceExplorerV2/Chat-Interface-GUI-Ollama-Py.git
cd Chat-Interface-GUI-Ollama-Py

# 2. Instala las dependencias
pip install -r requirements.txt

# 3. Inicia Ollama (en otra terminal)
ollama serve

# 4. Descarga un modelo si no tienes ninguno
ollama pull llama3.2

# 5. Corre la app
python main.py
```

### 🛠️ Stack tecnológico

| Herramienta | Para qué |
|---|---|
| `Python 3.10+` | Lenguaje principal |
| `CustomTkinter 5.2.2` | Widgets de UI modernos |
| `LangGraph 0.2.74` | Grafo de conversación y memoria |
| `LangChain Ollama` | Integración con modelos Ollama |
| `Requests` | Obtener modelos disponibles desde la API de Ollama |

---

<div align="center">

Made with 💜 by [JulianDataScienceExplorerV2](https://github.com/JulianDataScienceExplorerV2)

*Running local AI, the way it should be.*

</div>
