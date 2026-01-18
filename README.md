# Manga Ajuste Fino

- Python 3.11

## 🚀 Instalação

### 1. Crie ambiente virtual


**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Instale dependências

```bash
pip install -r requirements.txt
```

## 📖 Uso

```bash
python ajuste-fino.py
```

## Observação:

- Você precisa ter acesso ao modelo no Hugging Face (licença/credenciais se aplicável).
- Este script treina um adaptador LoRA. Depois você importa no Ollama via Modelfile + ADAPTER.
