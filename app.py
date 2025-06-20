from flask import Flask, render_template, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os
import time

app = Flask(__name__)

# Inisialisasi model
MODEL_LOADED = False
tokenizer = None
model = None

def load_model():
    global MODEL_LOADED, tokenizer, model
    
    try:
        model_path = os.path.join('models', 'best_washinbot_t5_model')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        print("Memuat tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        
        print("Memuat model...")
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        MODEL_LOADED = True
        print("Model berhasil dimuat!")
        
    except Exception as e:
        print(f"Gagal memuat model: {str(e)}")
        MODEL_LOADED = False

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# API endpoint untuk chat
@app.route('/api/chat', methods=['POST'])
def chat():
    start_time = time.time()
    
    data = request.get_json()
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'error': 'Pesan tidak boleh kosong'}), 400
    
    try:
        if not MODEL_LOADED:
            return jsonify({
                'response': "Maaf, model AI sedang dalam pemeliharaan. Silakan coba lagi nanti.",
                'processing_time': 0
            })
        
        # Preprocess input
        input_text = f"pertanyaan: {message} jawaban:"
        
        # Tokenisasi input
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        
        # Generate response
        output = model.generate(
            input_ids,
            max_length=500,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2,
            temperature=0.7
        )
        
        # Decode output
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        processing_time = round(time.time() - start_time, 2)
        
        return jsonify({
            'response': response,
            'processing_time': processing_time
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            'error': 'Terjadi kesalahan saat memproses permintaan Anda',
            'processing_time': 0
        }), 500

# Jalankan aplikasi
if __name__ == '__main__':
    print("Menyiapkan aplikasi...")
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)