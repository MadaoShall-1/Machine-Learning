"""
Simple Isolated Test
=====================
Test the detect_sentence_languages function in isolation
"""

from transformers import pipeline
import torch
from collections import Counter

print("="*70)
print("ISOLATED TEST: detect_sentence_languages")
print("="*70)

# Setup
if torch.cuda.is_available():
    device = 0
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    device = -1
    print(f"💻 CPU")

# Load model
print("\nLoading model...")
lang_id_model = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection",
    device=device
)
print("✅ Model loaded")

# Language mapping
lang_id_map = {
    'es': 'es',
    'en': 'en',
    'ar': 'ar',
    'hi': 'hi',
    'zh': 'zh',
}

# Test sentence
test_text = "El hueso fue descubierto hace aproximadamente 150 años. It's fascinating how these ancient creatures roamed."

print(f"\n📝 Test sentence:")
print(f"   {test_text[:80]}...")

# Replicate the function logic step by step
print(f"\n{'='*70}")
print("STEP-BY-STEP EXECUTION")
print(f"{'='*70}")

try:
    # Step 1: Split words
    words = test_text.split()
    print(f"\n✅ Step 1: Split into {len(words)} words")
    
    # Step 2: Create chunks
    chunk_size = 6 if device == 0 else 4
    chunks = []
    chunk_info = []
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        if len(chunk_text.strip()) >= 3:
            chunks.append(chunk_text)
            chunk_info.append((i, len(chunk_words)))
    
    print(f"✅ Step 2: Created {len(chunks)} chunks (chunk_size={chunk_size})")
    for idx, (chunk_text, (start, length)) in enumerate(zip(chunks[:5], chunk_info[:5])):
        print(f"   Chunk {idx}: '{chunk_text[:40]}...' (words {start}-{start+length})")
    
    # Step 3: Batch prediction
    print(f"\n🔄 Step 3: Running model on {len(chunks)} chunks...")
    predictions = lang_id_model(chunks)
    print(f"✅ Got {len(predictions)} predictions")
    
    # Step 4: Show predictions
    print(f"\n📊 Step 4: Predictions:")
    for idx, (chunk_text, pred) in enumerate(zip(chunks[:10], predictions[:10])):
        if isinstance(pred, list):
            pred = pred[0]
        print(f"   '{chunk_text[:35]}...' → {pred['label']} ({pred['score']:.3f})")
    
    # Step 5: Map to words
    word_langs = ['en'] * len(words)
    
    for (start_idx, chunk_len), pred in zip(chunk_info, predictions):
        if isinstance(pred, list):
            pred = pred[0]
        
        detected_code = pred.get('label', 'en')
        mapped_lang = lang_id_map.get(detected_code, 'en')
        
        for j in range(chunk_len):
            if start_idx + j < len(words):
                word_langs[start_idx + j] = mapped_lang
    
    print(f"\n✅ Step 5: Mapped to words")
    
    # Show results
    print(f"\n📋 Word-level results (first 20):")
    print(f"   {'Word':<25} {'Language':<10}")
    print(f"   {'-'*40}")
    for word, lang in zip(words[:20], word_langs[:20]):
        print(f"   {word:<25} {lang:<10}")
    
    # Count distribution
    lang_dist = Counter(word_langs)
    print(f"\n📊 Language distribution:")
    for lang, count in lang_dist.items():
        pct = count / len(words) * 100
        print(f"   {lang}: {count} words ({pct:.1f}%)")
    
    # Check if we have both languages
    if 'es' in lang_dist and 'en' in lang_dist:
        print(f"\n✅✅ SUCCESS! Both Spanish and English detected!")
        print(f"   This function DOES work correctly!")
    else:
        print(f"\n❌ PROBLEM! Only detected: {set(lang_dist.keys())}")
        print(f"   Model predictions might be wrong")
    
except Exception as e:
    print(f"\n❌ Error during execution: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
print("""
If you see "✅ SUCCESS! Both Spanish and English detected!":
   → The function DOES work in isolation
   → Problem must be in how it's called during processing
   
If you see only 'en' detected:
   → The model itself is not detecting Spanish correctly
   → Might need to adjust chunk_size or use different approach
""")
