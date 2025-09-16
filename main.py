import io
import os
import csv
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import (
    CLIPProcessor,
    CLIPModel,
    AutoTokenizer,
    AutoModelForCausalLM
)

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI()

# ----------------------------
# Load Medical Captions Dataset
# ----------------------------
def load_medical_captions_from_txt(txt_filepath):
    captions_dict = {}
    with open(txt_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split("\t", 1)  # split by TAB, keep caption
            if len(parts) < 2:
                continue
            image_name, caption = parts[0].strip(), parts[1].strip()
            if image_name not in captions_dict:
                captions_dict[image_name] = []
            captions_dict[image_name].append(caption)
    return captions_dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
caption_file_path = os.path.join(BASE_DIR, "captions", "captions_both.txt")
medical_captions = load_medical_captions_from_txt(caption_file_path)

# # Example usage
# caption_file_path = r"C:\Users\vasup\Downloads\archive (2)\all_data\train\both captions\captions_both.txt"
# medical_captions = load_medical_captions_from_txt(caption_file_path)

# Flatten captions into a single list
candidate_captions = []
for caps in medical_captions.values():
    candidate_captions.extend(caps)


print(f"Loaded {len(candidate_captions)} medical captions.")
if len(candidate_captions) == 0:
    raise ValueError("No captions loaded from file. Check file path and format.")

# ----------------------------
# Load Models
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_model_name = "openai/clip-vit-base-patch32"   # free and public
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
clip_model.to(device)
clip_model.eval()

# BioGPT (for paraphrasing into clinical language)
gpt_model_name = "microsoft/BioGPT-Large"
gpt_model = AutoModelForCausalLM.from_pretrained(gpt_model_name)
gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)
gpt_model.to(device)
gpt_model.eval()

# ----------------------------
# Precompute Caption Embeddings
# ----------------------------
def embed_captions(captions, batch_size=64):
    captions = [c for c in captions if isinstance(c, str) and c.strip()]
    if not captions:
        raise ValueError("No valid captions to embed.")
    embeddings = []
    max_len = clip_processor.tokenizer.model_max_length  # 77 for CLIP

    for i in range(0, len(captions), batch_size):
        batch = captions[i:i+batch_size]
        inputs = clip_processor(text=batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = clip_model.get_text_features(**inputs)
        batch_embeds = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        embeddings.append(batch_embeds.cpu())
    return torch.cat(embeddings)

caption_embeddings = embed_captions(candidate_captions)

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image

def embed_image(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.cpu()

def rank_captions(image_embedding, caption_embeddings, captions, top_k=5):
    similarities = torch.matmul(caption_embeddings, image_embedding.T).squeeze(1)
    values, indices = similarities.topk(top_k, largest=True)
    ranked_captions = [captions[idx] for idx in indices]
    return ranked_captions

def paraphrase_caption(caption, max_length=80):
    prompt = "Rewrite this finding in formal clinical language: " + caption
    inputs = gpt_tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = gpt_model.generate(
        inputs,
        max_length=inputs.shape[1] + 80,  # input length + desired output length
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    paraphrased = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return paraphrased.replace(prompt, "").strip()

# ----------------------------
# API Endpoint
# ----------------------------
@app.post("/caption/")
async def get_captions(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)
    image_emb = embed_image(image)

    # Retrieve top 3 candidate captions
    top_captions = rank_captions(image_emb, caption_embeddings, candidate_captions, top_k=3)

    # Refine with BioGPT
    refined_captions = [paraphrase_caption(c) for c in top_captions]

    return JSONResponse(content={"captions": refined_captions})

# ----------------------------
# Run with Uvicorn
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi.middleware.cors import CORSMiddleware

# # Allow Netlify frontend to access backend
# origins = [
#     "https://ai-medical-image-captioning-system.netlify.app",  # your Netlify frontend URL
#     # "http://localhost:3000", # optional, if testing locally
# ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,   # or ["*"] to allow all (for testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
