# Local LLM Chatbot GUI (Ollama + Python)

A beautiful, local-first graphical user interface (GUI) written in Python to chat with state-of-the-art open source Large Language Models (LLMs) running completely offline via Ollama. 

Una hermosa interfaz gráfica local (GUI) escrita en Python para chatear con LLMs open-source de última generación corriendo 100% offline a través de Ollama.

---

<p align="center">
  <img src="https://raw.githubusercontent.com/JulianDataScienceExplorerV2/Local-LLM-Chatbot-GUI-Ollama/main/assets/llm_chat_gui.png" alt="Local LLM Chatbot GUI" width="800"/>
</p>

## Features / Caracteristicas
- **100% Private**: Runs entirely on your local hardware. No API keys, no internet connection required.
- **Graphical Interface**: Custom desktop UI built using Python GUI libraries.
- **Model Switching**: Select between LLaMA 3, Mistral, Gemma, or any model installed locally.
- **Streaming Responses**: Real-time token generation for instant feedback.

## Tech Stack / Lenguajes
- **Language**: Python 3.10+
- **Backend API**: `requests` (connecting to localhost Ollama listener)
- **GUI Framework**: `Tkinter` (Customized Desktop Window Engine)
- **LLM Engine**: Ollama inference wrapper

## Requirements / Requisitos
1. You must have [Ollama](https://ollama.com/) installed and running locally.
2. Pull at least one model:
   ```bash
   ollama pull llama3
   ```

## How to Run / Como Ejecutar

```bash
# Clone the client
git clone https://github.com/JulianDataScienceExplorerV2/Local-LLM-Chatbot-GUI-Ollama.git
cd Local-LLM-Chatbot-GUI-Ollama

# Run the Chat interface
python app.py
```

---

<div align="center">
<b>Julian David Urrego Lancheros</b> <br>
<i>Generative AI & Python Developer</i>
</div>
