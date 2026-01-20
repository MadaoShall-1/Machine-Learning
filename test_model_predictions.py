"""
Debug: What Does The Model Actually Predict?
=============================================
This will show us exactly what the language ID model is detecting
"""

from transformers import pipeline
import torch

print("="*70)
print("TESTING LANGUAGE ID MODEL PREDICTIONS")
print("="*70)

# Load model
if torch.cuda.is_available():
    device = 0
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = -1
    print(f"Using CPU")

model = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection",
    device=device
)

print("✅ Model loaded\n")

# Test actual Spanish-English code-switched sentences
test_sentences = [
    "El hueso fue descubierto hace aproximadamente 150 años.",  # Pure Spanish
    "It's fascinating how these ancient creatures roamed the earth.",  # Pure English
    "El hueso fue descubierto. It's fascinating.",  # Code-switched
    "Parte de uno de los fósiles más grandes. Such discoveries help us.",  # Code-switched
]

print("Testing on real sentences:\n")

for sent in test_sentences:
    print(f"Text: {sent[:60]}...")
    
    # Full sentence detection
    result = model(sent)
    if isinstance(result, list):
        result = result[0]
    
    print(f"   Full sentence: {result['label']} (score: {result['score']:.3f})")
    
    # Chunk-level detection (like our code does)
    words = sent.split()
    chunk_size = 4
    
    print(f"   Chunks:")
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        chunk_text = ' '.join(chunk_words)
        
        chunk_result = model(chunk_text)
        if isinstance(chunk_result, list):
            chunk_result = chunk_result[0]
        
        print(f"      '{chunk_text[:40]}...' → {chunk_result['label']} ({chunk_result['score']:.2f})")
    
    print()

print("="*70)
print("ANALYSIS")
print("="*70)

print("""
Look at the predictions above:

✅ If you see 'es' for Spanish chunks and 'en' for English chunks:
   → Model is working correctly
   → Problem is in the token mapping logic

❌ If you see ONLY 'en' for everything:
   → Model is detecting everything as English
   → This is the root cause!

❌ If you see 'ca' (Catalan) or other unexpected codes:
   → Model might be confused
   → Need to adjust chunk size or add filtering
""")
