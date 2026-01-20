"""
Inspect Language Detection Cache
=================================
Check what languages were actually detected by the model
"""

import sys
sys.path.insert(0, '.')

from data_processing_phase2 import SwitchLinguaProcessor, ProcessingConfig
import pandas as pd
import ast

def inspect_detection_results():
    """Inspect what the model detected"""
    
    print("="*70)
    print("INSPECTING LANGUAGE DETECTION RESULTS")
    print("="*70)
    
    # Initialize processor
    config = ProcessingConfig()
    config.language_pairs = ['spanish_eng']
    processor = SwitchLinguaProcessor(config)
    
    # Load a few Spanish sentences
    csv_path = "data/Spanish_eng.csv"
    df = pd.read_csv(csv_path, nrows=5)
    
    print(f"\n✅ Loaded {len(df)} rows")
    
    # Extract first few sentences
    sentences = []
    for _, row in df.iterrows():
        result = ast.literal_eval(row['data_generation_result'])
        sentences.extend(result[:2])  # First 2 from each row
        if len(sentences) >= 5:
            break
    
    print(f"✅ Extracted {len(sentences)} test sentences\n")
    
    # Test each sentence
    for idx, sentence in enumerate(sentences[:3], 1):
        print(f"\n{'='*70}")
        print(f"SENTENCE {idx}")
        print(f"{'='*70}")
        print(f"\n📝 Text: {sentence[:100]}...")
        
        # Detect language pattern
        pattern = processor.lang_detector.detect_sentence_languages(sentence, 'spanish_eng')
        
        if pattern is None:
            print(f"\n❌ Pattern is None!")
            print(f"   Reasons:")
            print(f"   1. pair != 'spanish_eng': {('spanish_eng' != 'spanish_eng')}")
            print(f"   2. lang_id_available: {processor.lang_detector.lang_id_available}")
            continue
        
        print(f"\n✅ Pattern detected:")
        print(f"   Words: {len(pattern['words'])}")
        
        # Show word-by-word detection
        print(f"\n   {'Word':<30} {'Detected Language':<20}")
        print(f"   {'-'*50}")
        
        from collections import Counter
        lang_counts = Counter()
        
        for word, lang in zip(pattern['words'][:20], pattern['langs'][:20]):
            print(f"   {word:<30} {lang:<20}")
            lang_counts[lang] += 1
        
        print(f"\n   Summary for first 20 words:")
        for lang, count in lang_counts.items():
            print(f"      {lang}: {count} words ({count/20*100:.1f}%)")
        
        # Now tokenize this sentence
        print(f"\n   🔤 Tokenizing with language IDs...")
        
        sentence_meta = {
            'text': sentence,
            'first_language': 'Spanish',
            'second_language': 'English',
            'inferred_pair': 'spanish_eng',
            'original_idx': 0
        }
        
        tokenized = processor.tokenize_with_language_ids(sentence_meta)
        
        # Show token-level languages
        print(f"\n   Token-level language IDs (first 20):")
        print(f"   {'Idx':<5} {'Token':<20} {'LangID':<10}")
        print(f"   {'-'*50}")
        
        token_lang_counts = Counter()
        for i in range(min(20, len(tokenized['tokens']))):
            token = tokenized['tokens'][i]
            lang = tokenized['language_ids'][i]
            if lang != 'special':
                token_lang_counts[lang] += 1
            print(f"   {i:<5} {token:<20} {lang:<10}")
        
        print(f"\n   Token language distribution:")
        for lang, count in token_lang_counts.items():
            print(f"      {lang}: {count} tokens")
        
        # Generate labels and check switches
        labeled = processor.generate_predictive_labels(tokenized)
        switches = sum(1 for s in labeled['switch_labels'] if s == 1)
        
        print(f"\n   🔄 Switches in this sentence: {switches}")
        
        if switches == 0:
            print(f"\n   ❌ NO SWITCHES!")
            
            # Find the problem
            unique_token_langs = set(tokenized['language_ids']) - {'special'}
            print(f"   Unique token languages: {unique_token_langs}")
            
            if len(unique_token_langs) == 1:
                print(f"\n   🐛 ROOT CAUSE FOUND:")
                print(f"      All tokens are marked as: {unique_token_langs.pop()}")
                print(f"      The pattern detection works, but tokenization doesn't use it!")
        else:
            print(f"   ✅ Switches detected correctly!")


if __name__ == "__main__":
    inspect_detection_results()
