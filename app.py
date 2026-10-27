import streamlit as st
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

st.set_page_config(page_title="Shahajjo_Dicode AI", layout="centered")

st.title("💻 Shahajjo_Dicode - HTML/CSS Web Generator")
st.write("আপনার কাস্টম স্ক্র্যাচড মডেল দিয়ে প্রফেশনাল ওয়েবসাইট কোড তৈরি করুন!")

# Streamlit secrets থেকে অটো টোকেন রিড করবে
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception as e:
    HF_TOKEN = None

REPO_ID = "Vidclip/Shahajjo_Dicode"

@st.cache_resource
def load_model_and_tokenizer(token):
    if not token:
        return None, None, "HF_TOKEN not found in Streamlit Secrets!"
    
    try:
        # Hugging Face থেকে ফাইলগুলো ডাউনলোড করা
        tokenizer_path = hf_hub_download(repo_id=REPO_ID, filename="tokenizer.json", repo_type="model", token=token)
        model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pth", repo_type="model", token=token)
        
        # টোকেনাইজার লোড
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_path,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>"
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        # কাস্টম কজুয়াল ট্রান্সফর্মার আর্কিটেকচার
        class ScratchCausalTransformer(nn.Module):
            def __init__(self, vocab_size, d_model=256, n_heads=4, num_layers=4, max_len=256):
                super().__init__()
                self.token_embedding = nn.Embedding(vocab_size, d_model)
                self.position_embedding = nn.Embedding(max_len, d_model)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, 
                    nhead=n_heads, 
                    dim_feedforward=d_model * 4,
                    batch_first=True
                )
                self.transformer_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.fc_out = nn.Linear(d_model, vocab_size)
                self.max_len = max_len

            def forward(self, input_ids):
                batch_size, seq_len = input_ids.shape
                positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
                tok_emb = self.token_embedding(input_ids)
                pos_emb = self.position_embedding(positions)
                x = tok_emb + pos_emb
                mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(input_ids.device)
                x = self.transformer_blocks(x, mask=mask, is_causal=True)
                return self.fc_out(x)

        model = ScratchCausalTransformer(vocab_size=len(tokenizer), d_model=256, n_heads=4, num_layers=4, max_len=256)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        return model, tokenizer, "Success"
    except Exception as e:
        return None, None, str(e)

if HF_TOKEN:
    with st.spinner("Downloading model & tokenizer securely from Hugging Face..."):
        model, tokenizer, status = load_model_and_tokenizer(HF_TOKEN)
        
    if status == "Success":
        st.success("Model loaded successfully from Hugging Face!")
        
        user_prompt = st.text_area("Enter your prompt (e.g., Create an e-commerce product card for premium chocolate with price and buy button):")
        
        if st.button("Generate Code"):
            if user_prompt:
                with st.spinner("Generating HTML/CSS..."):
                    input_text = f"[PROMPT] {user_prompt} [CODE]"
                    input_ids = tokenizer.encode(input_text, return_tensors="pt")
                    
                    # জেনারেশন লজিক
                    with torch.no_grad():
                        for _ in range(450):
                            outputs = model(input_ids)
                            next_token_logits = outputs[:, -1, :]
                            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
                            
                            input_ids = torch.cat([input_ids, next_token], dim=1)
                            if next_token.item() == tokenizer.eos_token_id:
                                break
                                
                    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
                    
                    if "[CODE]" in generated_text:
                        code_output = generated_text.split("[CODE]")[-1].replace("</s>", "").strip()
                    else:
                        code_output = generated_text
                        
                    # টোকেনাইজারের 'Ġ' ক্যারেক্টার ক্লিন করার কোড
                    code_output = code_output.replace("Ġ", " ").replace("Ċ", "\n")
                        
                    st.subheader("Generated HTML/CSS Output:")
                    st.code(code_output, language="html")
            else:
                st.warning("Please enter a prompt first.")
    else:
        st.error(f"Error loading model: {status}")
else:
    st.error("HF_TOKEN is missing! Please configure it in Streamlit Secrets.")
