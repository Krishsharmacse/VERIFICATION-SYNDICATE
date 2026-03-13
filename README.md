# 🦅 Verification Syndicate: AI-Powered Misinformation Detection

A multi-agent system designed to identify, analyze, and educate against misinformation using LangGraph, Gemini 1.5, and real-time news verification.

## 🚀 Overview

Verification Syndicate is a robust, lightweight, and deployable misinformation detection system. It leverages a team of AI agents that collaborate to verify claims, analyze media (Multimodal), and provide educational context to the user.

### Key Features
- **🕵️ Multi-Agent Syndicate**: Uses specialized agents (Judge, Educator, Verification, Multi-modal) for comprehensive analysis.
- **🔍 Real-time Verification**: Integration with DuckDuckGo Search and APITube for live fact-checking.
- **📸 Multimodal Capabilities**: Handles Text, Images (OCR), and Audio (Transcription) using Gemini 1.5 Flash.
- **🧠 Enhanced Heuristics**: Custom linguistic forensics for detecting sensationalism, WhatsApp-style manipulation, and Hinglish misinformation.
- **⚡ Lightweight Prototype**: Designed to run without heavy local BERT models (~200MB instead of 2GB+).
- **🛡️ Robust Fallbacks**: Intelligent fallback to heuristic analysis if AI API keys are unavailable.

## 🛠️ Architecture

The system uses a **LangGraph** workflow to coordinate agents:
1. **Multimodal Layer**: Extracts content from text, images, or audio.
2. **Analysis Syndicate**: Concurrent evaluation by Deep Learning classifiers and Forensic linguistic agents.
3. **Verification Layer**: Live search and news ranking to cross-reference claims.
4. **Judge Agent**: Synthesizes all reports into a final verdict with a confidence score.
5. **Educator Agent**: Generates personalized educational content and counter-narratives.

## 🚦 Getting Started

### Prerequisites
- Python 3.10+
- [Gemini API Key](https://aistudio.google.com/) (Optional but recommended for full AI power)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Krishsharmacse/VERIFICATION-SYNDICATE.git
   cd VERIFICATION-SYNDICATE
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Configure your Environment:
   Create a `.env` file based on `.env.example`:
   ```env
   GEMINI_API_KEY=your_key_here
   APITUBE_API_KEY=your_key_here
   ```

### Running Locally
```powershell
.\.venv\Scripts\python main.py
```
The server will start at `http://localhost:8000`.

## 📂 Project Structure
- `Backend/`: Core logic, agents, and graph definitions.
- `static/`: Frontend interface.
- `main.py`: Entry point for the FastAPI server.

## 📄 License
This project is licensed under the MIT License.
