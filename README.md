
# Text-to-SQL (Phi-3.5 Mini Instruct)

**Overview**  
End-to-end **Text-to-SQL system** built by fine-tuning **Phi-3.5 Mini Instruct**, merging LoRA adapters, and quantizing with **llama.cpp** to reduce model size from ~7GB to **2.4GB**, enabling efficient local inference via a lightweight backend and UI.

This repository focuses on the **model lifecycle**: fine-tuning → merging → GGUF conversion → quantization → local serving.

---

## Model Details

- **Base model**: Phi-3.5 Mini Instruct  
- **Task**: Text-to-SQL generation  
- **Fine-tuning**: LoRA  
- **Inference format**: GGUF  
- **Quantization**: Q4 (llama.cpp)  
- **Final size**: ~2.4 GB  

---

## Workflow Summary

1. Fine-tune Phi-3.5 Mini Instruct using LoRA
2. Merge adapters into a single Hugging Face model
3. Convert merged model to GGUF format
4. Quantize using llama.cpp
5. Serve and test locally

---

### Clone the repository 

```bash
git clone https://github.com/sameerdev7/TextToSQL.git
```


### Install dependencies 

```bash
pip install -r requirements.txt
```


### Run LoRA fine-tuning

```bash
python lora_finetune.py
```

### Merge LoRA adapters with the base model

```bash
python merge_model.py
```

---

## GGUF Conversion and Quantization

### Download llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

### Convert merged model to GGUF

```bash
pip install -r requirements.txt

mkdir ../gguf

python convert_hf_to_gguf.py   --outfile ../gguf/phi-3.5-mini-sql-f16.gguf   --outtype f16   ../merged_model
```

### Build llama.cpp binaries

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build --parallel
```

### Quantize to 4-bit

```bash
build/bin/llama-quantize   ../gguf/phi-3.5-mini-sql-f16.gguf   ../gguf/phi-3.5-mini-sql-q4.gguf   Q4_K
```

---

## Test the Model

```bash
build/bin/llama-cli   -m ../gguf/phi-3.5-mini-sql-q4.gguf   -n 1000 -t 8   -p "Generate a SQL query for all users created last week."
```

## Start the LLM Inference Server 

```bash
./llama.cpp/build/bin/llama-server \
  -m models/phi35-text2sql-q4.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 2048 \
  -t 4 \
  --log-disable
```

## Start the Backend API 

```bash
cd backend
pip install -r requirements.txt
source venv/bin/activate
uvicorn api:app --reload
```

## Start the Frontend 

```
cd frontend 
pip install -r requirements.txt
source venv/bin/activate 
streamlit run app.py
```

---

## Notes

- The **GGUF model** is sufficient for inference.
- The merged `.safetensors` model is only required for re-quantization or further training.
- Designed for **local, low-latency inference**.

---

