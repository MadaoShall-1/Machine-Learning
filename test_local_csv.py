"""
Test Script for Local CSV Data
================================
This script tests the data processing pipeline on your local CSV files.
Run this BEFORE the full pipeline to catch any issues.

Usage:
    python test_local_csv.py

Expected file structure:
    data/
    ├── Spanish_eng.csv
    ├── Arabic_eng.csv
    ├── Hindi_eng.csv
    └── Chinese_eng.csv
"""

import pandas as pd
import ast
from pathlib import Path
from collections import Counter
from transformers import AutoTokenizer

def test_csv_loading():
    """Test loading CSV files"""
    print("="*70)
    print("TEST 1: CSV File Loading")
    print("="*70)
    
    csv_files = [
        "data/Spanish_eng.csv",
        "data/Arabic_eng.csv",
        "data/Hindi_eng.csv",
        "data/Chinese_eng.csv"
    ]
    
    for csv_file in csv_files:
        filepath = Path(csv_file)
        if filepath.exists():
            try:
                df = pd.read_csv(filepath, nrows=5)
                print(f"\n✅ {filepath.name}")
                print(f"   Rows (sample): {len(df)}")
                print(f"   Columns: {list(df.columns[:10])}")
            except Exception as e:
                print(f"\n❌ {filepath.name}: {e}")
        else:
            print(f"\n⚠️  {filepath.name}: File not found")
    
    return True


def test_data_extraction():
    """Test extracting sentences from data_generation_result"""
    print("\n" + "="*70)
    print("TEST 2: Data Extraction")
    print("="*70)
    
    csv_file = Path("data/Spanish_eng.csv")
    if not csv_file.exists():
        print("⚠️  Spanish_eng.csv not found, skipping test")
        return False
    
    df = pd.read_csv(csv_file, nrows=3)
    
    print(f"\nTesting on {len(df)} rows...")
    
    total_sentences = 0
    for idx, row in df.iterrows():
        result = row['data_generation_result']
        first_lang = row['first_language']
        second_lang = row['second_language']
        
        print(f"\nRow {idx}:")
        print(f"   first_language: {first_lang}")
        print(f"   second_language: {second_lang}")
        print(f"   data_generation_result type: {type(result)}")
        
        # Try to parse
        try:
            sentences = ast.literal_eval(result)
            print(f"   ✅ Parsed: {len(sentences)} sentences")
            total_sentences += len(sentences)
            
            # Show first sentence
            if len(sentences) > 0:
                print(f"   First sentence: {sentences[0][:100]}...")
        except Exception as e:
            print(f"   ❌ Parse error: {e}")
    
    print(f"\n✅ Total extracted: {total_sentences} sentences")
    return total_sentences > 0


def test_language_detection():
    """Test language detection on sample texts"""
    print("\n" + "="*70)
    print("TEST 3: Language Detection")
    print("="*70)
    
    from data_processing_phase2 import LanguageDetector
    
    detector = LanguageDetector()
    
    # Sample texts from your data
    test_cases = [
        ("El hueso fue descubierto", "latin", "Spanish"),
        ("It's fascinating", "latin", "English"),
        ("أنا تابعت خبر", "arabic", "Arabic"),
        ("महिलाओं की सेहत", "devanagari", "Hindi"),
        ("我今天去参观了", "chinese", "Chinese"),
    ]
    
    print("\nTesting script detection:\n")
    print(f"{'Text':<30} {'Expected':<12} {'Detected':<12} {'Status'}")
    print("-" * 70)
    
    all_correct = True
    for text, expected, description in test_cases:
        detected = detector.detect_script(text)
        status = "✅" if detected == expected else "⚠️"
        if detected != expected:
            all_correct = False
        print(f"{text:<30} {expected:<12} {detected:<12} {status}")
    
    return all_correct


def test_tokenization():
    """Test tokenization with XLM-RoBERTa"""
    print("\n" + "="*70)
    print("TEST 4: Tokenization")
    print("="*70)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        print("✅ Tokenizer loaded")
        
        # Test with a code-switched sentence
        test_text = "El hueso fue descubierto hace aproximadamente 150 años. It's fascinating."
        
        encoding = tokenizer(
            test_text,
            add_special_tokens=True,
            return_offsets_mapping=True
        )
        
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        
        print(f"\nTest text: {test_text}")
        print(f"Tokens ({len(tokens)}): {tokens[:15]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenization error: {e}")
        return False


def test_full_processing_sample():
    """Test the full processing on a small sample"""
    print("\n" + "="*70)
    print("TEST 5: Full Processing Sample")
    print("="*70)
    
    try:
        from data_processing_phase2 import SwitchLinguaProcessor, ProcessingConfig
        
        # Create config for CSV files
        config = ProcessingConfig()
        config.language_pairs = ["spanish_eng"]  # Test one pair first
        
        # Create processor
        processor = SwitchLinguaProcessor(config)
        
        # Load and process small sample
        csv_path = Path("data/Spanish_eng.csv")
        if not csv_path.exists():
            print("⚠️  Spanish_eng.csv not found")
            return False
        
        df = pd.read_csv(csv_path, nrows=2)
        print(f"✅ Loaded {len(df)} rows for testing")
        
        # Extract sentences
        sentences = processor.extract_sentences(df)
        print(f"✅ Extracted {len(sentences)} sentences")
        
        if len(sentences) > 0:
            # Try tokenizing one
            tokenized = processor.tokenize_with_language_ids(sentences[0])
            print(f"✅ Tokenized: {tokenized['length']} tokens")
            
            # Try generating labels
            processed = processor.generate_predictive_labels(tokenized)
            print(f"✅ Generated labels")
            
            # Validate
            is_valid = processor.validate_labels(processed)
            print(f"✅ Validation: {'PASSED' if is_valid else 'FAILED'}")
            
            return is_valid
        
        return False
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("LOCAL CSV DATA TESTING")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("CSV Loading", test_csv_loading()))
    results.append(("Data Extraction", test_data_extraction()))
    results.append(("Language Detection", test_language_detection()))
    results.append(("Tokenization", test_tokenization()))
    results.append(("Full Processing Sample", test_full_processing_sample()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<30} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
        print("\nYou're ready to run the full pipeline:")
        print("   python data_processing_phase2.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
