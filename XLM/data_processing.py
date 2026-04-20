import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
import json
import pickle
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ProcessingConfig:
    """Configuration for data processing"""
    # Dataset configuration
    dataset_name: str = "Shelton1013/SwitchLingua_text"
    language_pairs: List[str] = field(default_factory=lambda: [
        "spanish_eng",
        "arabic_eng", 
        "hindi_eng",
        "chinese_eng"
    ])
    
    # Directory paths
    output_dir: Path = Path("./processed_data")
    cache_dir: Path = Path("./cache")
    
    # Model configuration
    model_name: str = "xlm-roberta-base"  # Can use "bert-base-multilingual-cased"
    max_length: int = 512
    spanish_lid_backend: str = "codeswitch"
    
    # Duration binning thresholds (in tokens)
    duration_bins: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "Small": (1, 2),      # Class 0: 1-2 tokens (lexical insertions)
        "Medium": (3, 6),     # Class 1: 3-6 tokens (phrase-level)
        "Large": (7, 999)     # Class 2: 7+ tokens (clausal/sentence)
    })
    
    # Processing parameters
    min_sentence_length: int = 5  # Minimum tokens per sentence
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Language Detection Utilities
# ============================================================================

class LanguageDetector:
    """Detect language at character/token level for code-switching"""

    def __init__(
        self,
        spanish_lid_backend: str = "codeswitch",
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        self.spanish_lid_backend = spanish_lid_backend
        self.model_cache_dir = Path(cache_dir) if cache_dir is not None else Path.cwd() / "cache" / "hf_models"
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(self.model_cache_dir)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(self.model_cache_dir / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(self.model_cache_dir / "transformers")
        self.script_ranges = {
            'arabic': [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)],
            'devanagari': [(0x0900, 0x097F)],
            'chinese': [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
            'latin': [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x00FF)]
        }

        self.lang_id_map = {
            'es': 'es',
            'spa': 'es',
            'en': 'en',
            'eng': 'en',
            'ar': 'ar',
            'hi': 'hi',
            'zh': 'zh',
        }

        try:
            from transformers import pipeline, AutoModelForTokenClassification

            device = -1
            print(f"   Using CPU for language detection backend: {self.spanish_lid_backend}")

            if self.spanish_lid_backend == 'codeswitch':
                self.codeswitch_tokenizer = AutoTokenizer.from_pretrained(
                    'sagorsarker/codeswitch-spaeng-lid-lince',
                    cache_dir=str(self.model_cache_dir),
                )
                self.codeswitch_model = AutoModelForTokenClassification.from_pretrained(
                    'sagorsarker/codeswitch-spaeng-lid-lince',
                    cache_dir=str(self.model_cache_dir),
                )
                self.codeswitch_model.eval()
                self.lang_id_model = None
            else:
                self.lang_id_model = pipeline(
                    'text-classification',
                    model='papluca/xlm-roberta-base-language-detection',
                    device=device,
                    batch_size=8,
                    model_kwargs={"cache_dir": str(self.model_cache_dir)},
                )

            self.lang_id_available = True
            self.device = device
            print('   Language ID model loaded successfully')
        except Exception as e:
            self.lang_id_available = False
            self.device = -1
            print(f"   Warning: Could not load language ID model: {e}")
            print('   Will use script-based detection')

        self.prediction_cache = {}
        self.fallback_stats = Counter()

    def detect_script(self, text: str) -> str:
        """Detect dominant script in text (fast)"""
        if not text or not text.strip():
            return 'unknown'

        script_counts = Counter()
        for char in text:
            code_point = ord(char)
            for script, ranges in self.script_ranges.items():
                if any(start <= code_point <= end for start, end in ranges):
                    script_counts[script] += 1
                    break

        if not script_counts:
            return 'latin'

        return script_counts.most_common(1)[0][0]

    def detect_sentence_languages(self, full_text: str, pair: str) -> dict:
        """Detect a word-level language sequence for Spanish-English sentences."""
        if pair != 'spanish_eng' or not self.lang_id_available:
            return None

        cache_key = (self.spanish_lid_backend, full_text)
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]

        try:
            if self.spanish_lid_backend == 'codeswitch':
                result = self._detect_sentence_languages_codeswitch(full_text)
            else:
                result = self._detect_sentence_languages_papluca(full_text)
            self.prediction_cache[cache_key] = result
            return result
        except Exception as e:
            print(f"\n   Warning: Language detection error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _detect_sentence_languages_papluca(self, full_text: str) -> Optional[dict]:
        words = full_text.split()
        if len(words) == 0:
            return None

        window_radius = 2
        contexts = []
        for idx in range(len(words)):
            start_idx = max(0, idx - window_radius)
            end_idx = min(len(words), idx + window_radius + 1)
            contexts.append(' '.join(words[start_idx:end_idx]))

        predictions = self.lang_id_model(contexts)
        word_langs = []
        for pred in predictions:
            if isinstance(pred, list):
                pred = pred[0]
            detected_code = str(pred.get('label', 'en')).lower()
            word_langs.append(self.lang_id_map.get(detected_code, 'en'))

        return {'words': words, 'langs': word_langs}

    def _detect_sentence_languages_codeswitch(self, full_text: str) -> Optional[dict]:
        words = full_text.split()
        if len(words) == 0:
            return None

        encoding = self.codeswitch_tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',
            truncation=True,
        )

        with torch.no_grad():
            outputs = self.codeswitch_model(**encoding)
            pred_ids = outputs.logits.argmax(dim=-1)[0].tolist()

        word_ids = encoding.word_ids(batch_index=0)
        vote_map = defaultdict(list)
        for pred_id, word_id in zip(pred_ids, word_ids):
            if word_id is None:
                continue
            label = str(self.codeswitch_model.config.id2label[pred_id]).lower()
            vote_map[word_id].append(self.lang_id_map.get(label, 'en'))

        word_langs = []
        for word_idx in range(len(words)):
            votes = vote_map.get(word_idx, ['en'])
            word_langs.append(Counter(votes).most_common(1)[0][0])

        return {'words': words, 'langs': word_langs}

    def get_language_from_script(self, script: str, pair: str) -> str:
        """Map script to language (fallback for non-Latin pairs)"""
        script_to_lang = {
            'arabic': 'ar',
            'devanagari': 'hi',
            'chinese': 'zh',
            'latin': 'en'
        }
        return script_to_lang.get(script, 'en')

    def detect_token_language(self, token_text: str, full_sentence: str,
                             token_idx: int, pair: str,
                             sentence_pattern: dict = None) -> str:
        """Fallback token-level language detection when word-level mapping is unavailable."""
        script = self.detect_script(token_text)
        if script != 'latin':
            return self.get_language_from_script(script, pair)

        if pair == 'spanish_eng' and sentence_pattern is not None:
            words = sentence_pattern['words']
            langs = sentence_pattern['langs']
            token_clean = token_text.lower().lstrip("\u2581").strip(" ,!?;:'\"")
            if not token_clean:
                self.fallback_stats['empty_token_clean'] += 1
                return 'en'

            for i, word in enumerate(words):
                word_lower = word.lower()
                if token_clean in word_lower or word_lower in token_clean:
                    self.fallback_stats['substring_match'] += 1
                    return langs[i] if i < len(langs) else 'en'

            if len(token_clean) <= 2:
                lang_dist = Counter(langs)
                if lang_dist:
                    self.fallback_stats['short_token_majority'] += 1
                    return lang_dist.most_common(1)[0][0]

            spanish_tokens = {
                'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'al',
                'es', 'son', 'fue', 'hace', 'anos', 'este', 'esta',
                'hueso', 'fosiles', 'ciudad', 'proyecto', 'descubierto'
            }
            english_tokens = {
                'the', 'is', 'are', 'it', 'its', "it's", 'this', 'that',
                'and', 'or', 'to', 'of', 'in', 'on', 'how', 'can',
                'fascinating', 'ancient', 'creatures', 'earth'
            }

            if token_clean in spanish_tokens:
                self.fallback_stats['heuristic_spanish'] += 1
                return 'es'
            elif token_clean in english_tokens:
                self.fallback_stats['heuristic_english'] += 1
                return 'en'

            lang_dist = Counter(langs)
            if lang_dist:
                self.fallback_stats['majority_fallback'] += 1
                return lang_dist.most_common(1)[0][0]

            self.fallback_stats['final_default_en'] += 1
            return 'en'

        return self.get_language_from_script(script, pair)


# ============================================================================
# Data Processor
# ============================================================================

class SwitchLinguaProcessor:
    """Process SwitchLingua dataset for streaming prediction"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.lang_detector = LanguageDetector(
            config.spanish_lid_backend,
            cache_dir=config.cache_dir / "hf_models" / config.spanish_lid_backend,
        )
        self.stats = defaultdict(lambda: defaultdict(int))
        
        print(f"✅ Initialized with tokenizer: {config.model_name}")
        print(f"   Vocab size: {len(self.tokenizer)}")
    
    def _resolve_local_csv(self, language_pair: str) -> Optional[Path]:
        """Resolve local CSV regardless of filename capitalization."""
        data_dir = Path("data")
        candidates = [
            data_dir / f"{language_pair}.csv",
            data_dir / f"{language_pair.capitalize()}.csv",
        ]
        if "_" in language_pair:
            first, second = language_pair.split("_", 1)
            candidates.append(data_dir / f"{first.capitalize()}_{second}.csv")

        for candidate in candidates:
            if candidate.exists():
                return candidate

        normalized_target = f"{language_pair}.csv".lower()
        for csv_path in data_dir.glob("*.csv"):
            if csv_path.name.lower() == normalized_target:
                return csv_path

        return None

    def load_raw_data(self, language_pair: str) -> pd.DataFrame:
        """Load data for a specific language pair from local CSV or Hugging Face"""
        print(f"\n📥 Loading {language_pair}...")
        
        # Try local CSV first
        local_csv_path = self._resolve_local_csv(language_pair)
        if local_csv_path is not None:
            try:
                df = pd.read_csv(local_csv_path)
                print(f"   ✅ Loaded from local CSV: {len(df)} samples")
                return df
            except Exception as e:
                print(f"   ⚠️  Error loading local CSV: {e}")
        
        # Fallback to Hugging Face
        try:
            from datasets import load_dataset
            print(f"   Attempting to load from Hugging Face...")
            dataset = load_dataset(
                self.config.dataset_name,
                language_pair,
                split='train'
            )
            df = pd.DataFrame(dataset)
            print(f"   ✅ Loaded from Hugging Face: {len(df)} samples")
            return df
        except Exception as e:
            print(f"   ❌ Error loading from Hugging Face: {e}")
            return pd.DataFrame()
    
    def infer_language_pair(self, first_lang: str, second_lang: str) -> str:
        """Infer language pair identifier from first and second language"""
        # Normalize language names to match our expected pairs
        lang_map = {
            'Spanish': 'spanish',
            'Arabic': 'arabic',
            'Hindi': 'hindi',
            'Chinese': 'chinese',
            'Mandarin': 'chinese',
            'English': 'eng'
        }
        
        first = lang_map.get(first_lang, first_lang.lower())
        second = lang_map.get(second_lang, second_lang.lower())
        
        # Ensure English is always second for consistency
        if first == 'eng':
            first, second = second, first
        
        return f"{first}_{second}"
    
    def extract_sentences(self, df: pd.DataFrame) -> List[Dict]:
        """Extract all code-switched sentences with metadata from dataset"""
        import ast
        
        sentences_with_meta = []
        
        for idx, row in df.iterrows():
            # Extract sentences from data_generation_result
            result = row.get('data_generation_result', [])
            first_lang = row.get('first_language', 'Unknown')
            second_lang = row.get('second_language', 'Unknown')
            
            # Infer language pair
            inferred_pair = self.infer_language_pair(first_lang, second_lang)
            
            sentence_list = []
            
            # Handle different data formats
            if isinstance(result, list):
                # Already a list
                sentence_list = [s for s in result if isinstance(s, str) and len(s.strip()) > 0]
            elif isinstance(result, str):
                # String representation of a list - use ast.literal_eval for safety
                try:
                    result_list = ast.literal_eval(result)
                    if isinstance(result_list, list):
                        sentence_list = [s for s in result_list if isinstance(s, str) and len(s.strip()) > 0]
                    else:
                        # Single string
                        if len(result.strip()) > 0:
                            sentence_list = [result]
                except (ValueError, SyntaxError):
                    # If parsing fails, treat as single sentence
                    if len(result.strip()) > 0:
                        sentence_list = [result]
            
            # Add each sentence with metadata
            for sent in sentence_list:
                sentences_with_meta.append({
                    'text': sent,
                    'first_language': first_lang,
                    'second_language': second_lang,
                    'inferred_pair': inferred_pair,
                    'original_idx': idx
                })
        
        print(f"   📝 Extracted {len(sentences_with_meta)} sentences")
        
        # Show language pair distribution
        pair_counts = Counter([s['inferred_pair'] for s in sentences_with_meta])
        print(f"   📊 Language pair distribution:")
        for pair, count in pair_counts.most_common():
            print(f"      - {pair}: {count} sentences")
        
        return sentences_with_meta
    

    def tokenize_with_language_ids(self, sentence_meta: Dict) -> Dict:
        """
        Tokenize text and assign language ID to each token.

        For Spanish-English, we first infer word-level language labels and then
        project them back to subword tokens using tokenizer word ids. This avoids
        substring heuristics and aligns training labels with word boundaries.
        """
        text = sentence_meta['text']
        pair = sentence_meta['inferred_pair']

        sentence_pattern = self.lang_detector.detect_sentence_languages(text, pair)
        words = text.split()
        tokenize_input = words if pair == 'spanish_eng' and words else text
        tokenize_kwargs = {
            'add_special_tokens': True,
            'return_offsets_mapping': True,
            'return_special_tokens_mask': True,
        }
        if pair == 'spanish_eng' and words:
            tokenize_kwargs['is_split_into_words'] = True

        encoding = self.tokenizer(tokenize_input, **tokenize_kwargs)

        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        token_ids = encoding['input_ids']
        offsets = encoding['offset_mapping']
        special_tokens_mask = encoding['special_tokens_mask']
        word_ids = encoding.word_ids() if hasattr(encoding, 'word_ids') else [None] * len(token_ids)

        language_ids = []
        word_boundaries = []

        for i, (token, (start_offset, end_offset), is_special) in enumerate(zip(tokens, offsets, special_tokens_mask)):
            if is_special:
                language_ids.append('special')
                word_boundaries.append(False)
                continue

            word_id = word_ids[i]
            if pair == 'spanish_eng' and sentence_pattern is not None and word_id is not None:
                lang = sentence_pattern['langs'][word_id]
                prev_word_id = word_ids[i - 1] if i > 0 else None
                is_word_start = word_id != prev_word_id
            else:
                token_text = text[start_offset:end_offset]
                lang = self.lang_detector.detect_token_language(token_text, text, i, pair, sentence_pattern)
                is_word_start = token.startswith('?') or (i == 0)

            language_ids.append(lang)
            word_boundaries.append(is_word_start)

        return {
            'text': text,
            'tokens': tokens,
            'token_ids': token_ids,
            'language_ids': language_ids,
            'word_boundaries': word_boundaries,
            'length': len(tokens),
            'pair': pair,
            'first_language': sentence_meta['first_language'],
            'second_language': sentence_meta['second_language']
        }

    def generate_predictive_labels(self, tokenized_data: Dict) -> Dict:
        """
        Generate labels for streaming prediction task.
        For each token t, predict:
        - ysw(t): whether language switches at t+1
        - ydur(t): duration of the switch (if any)
        """
        tokens = tokenized_data['tokens']
        lang_ids = tokenized_data['language_ids']
        n_tokens = len(tokens)
        
        # Initialize labels
        switch_labels = []  # Binary: 1 if next token switches language
        duration_labels = []  # Class ID (0/1/2) or -1 (ignore)
        
        for t in range(n_tokens - 1):  # Stop at n-1 since we predict t+1
            current_lang = lang_ids[t]
            next_lang = lang_ids[t + 1]
            
            # Skip special tokens
            if current_lang == 'special' or next_lang == 'special':
                switch_labels.append(-1)  # Ignore
                duration_labels.append(-1)
                continue
            
            # Check if switch occurs
            is_switch = (current_lang != next_lang)
            switch_labels.append(1 if is_switch else 0)
            
            # If switch, calculate duration
            if is_switch:
                # Count consecutive tokens in the new language
                burst_length = 1
                for j in range(t + 2, n_tokens):
                    if lang_ids[j] == next_lang and lang_ids[j] != 'special':
                        burst_length += 1
                    elif lang_ids[j] != 'special':
                        break
                
                # Bin the duration
                if 1 <= burst_length <= 2:
                    duration_class = 0  # Small
                elif 3 <= burst_length <= 6:
                    duration_class = 1  # Medium
                else:
                    duration_class = 2  # Large
                
                duration_labels.append(duration_class)
                
                # Update statistics
                self.stats['duration_distribution'][f'class_{duration_class}'] += 1
            else:
                duration_labels.append(-1)  # Ignore when no switch
        
        # Add dummy label for last token (no prediction needed)
        switch_labels.append(-1)
        duration_labels.append(-1)
        
        return {
            **tokenized_data,
            'switch_labels': switch_labels,
            'duration_labels': duration_labels
        }
    
    def validate_labels(self, processed_data: Dict) -> bool:
        """Validate label generation correctness"""
        tokens = processed_data['tokens']
        lang_ids = processed_data['language_ids']
        switch_labels = processed_data['switch_labels']
        duration_labels = processed_data['duration_labels']
        
        errors = []
        
        for t in range(len(tokens) - 1):
            if lang_ids[t] == 'special' or lang_ids[t+1] == 'special':
                continue
            
            actual_switch = (lang_ids[t] != lang_ids[t+1])
            predicted_switch = (switch_labels[t] == 1)
            
            # Check switch label consistency
            if actual_switch != predicted_switch:
                errors.append(f"Token {t}: Switch mismatch")
            
            # Check duration label logic
            if predicted_switch and duration_labels[t] == -1:
                errors.append(f"Token {t}: Switch without duration")
            elif not predicted_switch and duration_labels[t] != -1:
                errors.append(f"Token {t}: Duration without switch")
        
        if errors:
            print(f"   ⚠️  Validation errors: {len(errors)}")
            for err in errors[:5]:
                print(f"      - {err}")
            return False
        
        return True
    
    def process_language_pair(self, pair: str) -> List[Dict]:
        """Process all sentences for a language pair"""
        print(f"\n{'='*60}")
        print(f"Processing: {pair}")
        print(f"{'='*60}")
        
        # Load data
        df = self.load_raw_data(pair)
        if df.empty:
            return []
        
        # Extract sentences with metadata
        sentences_with_meta = self.extract_sentences(df)
        
        # Filter for the current pair (in case dataset has mixed pairs)
        sentences_for_pair = [s for s in sentences_with_meta if s['inferred_pair'] == pair]
        
        if not sentences_for_pair:
            print(f"   ⚠️  No sentences found for {pair}")
            return []
        
        print(f"   📝 Processing {len(sentences_for_pair)} sentences for {pair}")
        
        # OPTIMIZATION: Pre-compute language patterns for all sentences (batch!)
        # This is MUCH faster than doing it one-by-one during tokenization
        if pair == 'spanish_eng' and hasattr(self.lang_detector, 'lang_id_available') and self.lang_detector.lang_id_available:
            print(f"   🚀 Pre-computing language patterns (GPU-accelerated)...")
            
            # Batch process all sentences
            all_texts = [s['text'] for s in sentences_for_pair]
            batch_size_precompute = 64 if self.lang_detector.device == 0 else 32
            
            # Use tqdm for progress bar
            from tqdm import tqdm as progress_bar
            for i in progress_bar(range(0, len(all_texts), batch_size_precompute), desc="Language detection"):
                batch_texts = all_texts[i:i+batch_size_precompute]
                
                # Pre-compute for this batch
                for text in batch_texts:
                    _ = self.lang_detector.detect_sentence_languages(text, pair)
            
            print(f"   ✅ Pre-computed {len(all_texts)} sentence patterns")
        
        # Process each sentence (now with cached patterns!)
        processed_samples = []
        validation_passed = 0
        validation_failed = 0
        
        # Use tqdm again for tokenization progress
        from tqdm import tqdm as progress_bar
        for sent_idx, sentence_meta in enumerate(progress_bar(sentences_for_pair, desc=f"Tokenizing {pair}")):
            try:
                # Tokenize with language IDs (will use cached patterns)
                tokenized = self.tokenize_with_language_ids(sentence_meta)
                
                # Skip very short sentences
                if tokenized['length'] < self.config.min_sentence_length:
                    continue
                
                # Generate predictive labels
                processed = self.generate_predictive_labels(tokenized)
                
                # Validate
                if self.validate_labels(processed):
                    processed['sentence_idx'] = sent_idx
                    processed['original_idx'] = sentence_meta['original_idx']
                    processed_samples.append(processed)
                    validation_passed += 1
                else:
                    validation_failed += 1
                
                # Update statistics
                self.stats[pair]['total_sentences'] += 1
                self.stats[pair]['total_tokens'] += tokenized['length']
                self.stats[pair]['total_switches'] += sum(1 for x in processed['switch_labels'] if x == 1)
                
            except Exception as e:
                print(f"   ❌ Error processing sentence {sent_idx}: {e}")
                continue
        
        print(f"\n✅ Processed {pair}:")
        print(f"   - Valid samples: {validation_passed}")
        print(f"   - Failed validation: {validation_failed}")
        print(f"   - Total tokens: {self.stats[pair]['total_tokens']}")
        print(f"   - Total switches: {self.stats[pair]['total_switches']}")
        
        return processed_samples
    
    def create_splits(self, all_samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Split data into train/val/test with group-aware, pair-stratified sampling."""
        rng = np.random.default_rng(self.config.random_seed)
        grouped_samples = defaultdict(list)

        for sample in all_samples:
            group_key = (sample['pair'], sample.get('original_idx', sample.get('sentence_idx', -1)))
            grouped_samples[group_key].append(sample)

        grouped_keys_by_pair = defaultdict(list)
        for group_key in grouped_samples:
            grouped_keys_by_pair[group_key[0]].append(group_key)

        splits = {'train': [], 'val': [], 'test': []}

        for pair, group_keys in grouped_keys_by_pair.items():
            shuffled_keys = list(group_keys)
            rng.shuffle(shuffled_keys)

            total_groups = len(shuffled_keys)
            n_train = int(total_groups * self.config.train_split)
            n_val = int(total_groups * self.config.val_split)

            split_keys = {
                'train': shuffled_keys[:n_train],
                'val': shuffled_keys[n_train:n_train + n_val],
                'test': shuffled_keys[n_train + n_val:],
            }

            for split_name, keys in split_keys.items():
                for key in keys:
                    splits[split_name].extend(grouped_samples[key])
        
        print(f"\n📊 Data splits:")
        for split_name, split_data in splits.items():
            print(f"   - {split_name}: {len(split_data)} samples")
        
        return splits
    
    def save_processed_data(self, splits: Dict[str, List[Dict]]):
        """Save processed data to disk"""
        for split_name, split_data in splits.items():
            output_path = self.config.output_dir / f"{split_name}.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(split_data, f)
            print(f"   💾 Saved {split_name} to {output_path}")
        
        # Save statistics
        stats_path = self.config.output_dir / "statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(dict(self.stats), f, indent=2)
        print(f"   💾 Saved statistics to {stats_path}")
        
        # Save config
        config_path = self.config.output_dir / "config.json"
        config_dict = {
            'model_name': self.config.model_name,
            'language_pairs': self.config.language_pairs,
            'duration_bins': self.config.duration_bins,
            'max_length': self.config.max_length,
            'pad_token_id': self.tokenizer.pad_token_id,
            'padding_side': self.tokenizer.padding_side,
            'spanish_lid_backend': self.config.spanish_lid_backend,
            'processor_config': {
                **asdict(self.config),
                'output_dir': str(self.config.output_dir),
                'cache_dir': str(self.config.cache_dir),
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"   💾 Saved config to {config_path}")
    
    def run_complete_pipeline(self):
        """Execute the complete processing pipeline"""
        print("\n" + "="*60)
        print("PHASE 2: DATA PROCESSING PIPELINE")
        print("="*60)
        
        all_processed = []
        
        # Track statistics during processing
        pair_stats = defaultdict(lambda: {
            'samples': 0,
            'total_tokens': 0,
            'total_switches': 0,
            'total_positions': 0,
            'duration_counts': Counter()
        })
        
        # Process each language pair
        for pair in self.config.language_pairs:
            processed = self.process_language_pair(pair)
            all_processed.extend(processed)
            
            # Collect statistics for this pair
            for sample in processed:
                stats = pair_stats[pair]
                stats['samples'] += 1
                stats['total_tokens'] += len(sample['tokens'])
                
                # Count switches and positions
                for sw_label in sample['switch_labels']:
                    if sw_label != -1:
                        stats['total_positions'] += 1
                        if sw_label == 1:
                            stats['total_switches'] += 1
                
                # Count durations
                for dur_label in sample['duration_labels']:
                    if dur_label in [0, 1, 2]:
                        stats['duration_counts'][dur_label] += 1
        
        print(f"\n✅ Total processed samples: {len(all_processed)}")
        
        # ======================================================================
        # COMPREHENSIVE STATISTICS REPORT
        # ======================================================================
        
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE DATA STATISTICS")
        print("="*60)
        
        # 1. SAMPLES PER LANGUAGE PAIR
        print(f"\n{'='*60}")
        print("1️⃣  SAMPLES PER LANGUAGE PAIR (ROWS)")
        print(f"{'='*60}")
        
        total_samples = sum(s['samples'] for s in pair_stats.values())
        
        print(f"\n{'Language Pair':<20} {'Samples':<12} {'%':<8}")
        print(f"{'-'*60}")
        
        for pair in sorted(pair_stats.keys()):
            count = pair_stats[pair]['samples']
            pct = count / total_samples * 100 if total_samples > 0 else 0
            print(f"{pair:<20} {count:>11,} {pct:>6.2f}%")
        
        print(f"{'-'*60}")
        print(f"{'TOTAL':<20} {total_samples:>11,} {'100.00%':>7}")
        
        # 2. TOKEN STATISTICS
        print(f"\n{'='*60}")
        print("2️⃣  AVERAGE TOKEN COUNTS")
        print(f"{'='*60}")
        
        print(f"\n{'Language Pair':<20} {'Total Tokens':<15} {'Avg Tokens':<12}")
        print(f"{'-'*60}")
        
        total_all_tokens = 0
        for pair in sorted(pair_stats.keys()):
            stats = pair_stats[pair]
            total_tokens = stats['total_tokens']
            avg_tokens = total_tokens / stats['samples'] if stats['samples'] > 0 else 0
            total_all_tokens += total_tokens
            print(f"{pair:<20} {total_tokens:>14,} {avg_tokens:>10.2f}")
        
        avg_overall = total_all_tokens / total_samples if total_samples > 0 else 0
        print(f"{'-'*60}")
        print(f"{'OVERALL':<20} {total_all_tokens:>14,} {avg_overall:>10.2f}")
        
        # 3. SWITCH EVENT RATIO
        print(f"\n{'='*60}")
        print("3️⃣  SWITCH EVENT RATIO (y_sw = 1)")
        print(f"{'='*60}")
        
        print(f"\n{'Language Pair':<20} {'Switches':<11} {'Positions':<11} {'Rate %':<10}")
        print(f"{'-'*60}")
        
        total_all_switches = 0
        total_all_positions = 0
        
        for pair in sorted(pair_stats.keys()):
            stats = pair_stats[pair]
            switches = stats['total_switches']
            positions = stats['total_positions']
            rate = switches / positions * 100 if positions > 0 else 0
            
            total_all_switches += switches
            total_all_positions += positions
            
            print(f"{pair:<20} {switches:>10,} {positions:>10,} {rate:>8.4f}%")
        
        overall_rate = total_all_switches / total_all_positions * 100 if total_all_positions > 0 else 0
        print(f"{'-'*60}")
        print(f"{'OVERALL':<20} {total_all_switches:>10,} {total_all_positions:>10,} {overall_rate:>8.4f}%")
        
        # 4. DURATION DISTRIBUTION
        print(f"\n{'='*60}")
        print("4️⃣  DURATION DISTRIBUTION (Small/Medium/Large)")
        print(f"{'='*60}")
        
        print(f"\n📚 Definitions:")
        print(f"   Class 0 (Small):  1-2 tokens   (lexical insertions)")
        print(f"   Class 1 (Medium): 3-6 tokens   (phrase-level)")
        print(f"   Class 2 (Large):  7+ tokens    (clausal/sentential)")
        
        print(f"\n{'Language Pair':<20} {'Small':<10} {'Medium':<10} {'Large':<10}")
        print(f"{'-'*60}")
        
        total_dur_counts = Counter()
        
        for pair in sorted(pair_stats.keys()):
            stats = pair_stats[pair]
            dur_counts = stats['duration_counts']
            
            small = dur_counts[0]
            medium = dur_counts[1]
            large = dur_counts[2]
            
            total_dur_counts[0] += small
            total_dur_counts[1] += medium
            total_dur_counts[2] += large
            
            print(f"{pair:<20} {small:>9,} {medium:>9,} {large:>9,}")
        
        print(f"{'-'*60}")
        print(f"{'TOTAL':<20} {total_dur_counts[0]:>9,} {total_dur_counts[1]:>9,} {total_dur_counts[2]:>9,}")
        
        # Duration percentages
        print(f"\n{'Language Pair':<20} {'Small %':<10} {'Med %':<10} {'Large %':<10}")
        print(f"{'-'*60}")
        
        for pair in sorted(pair_stats.keys()):
            stats = pair_stats[pair]
            dur_counts = stats['duration_counts']
            total_dur = sum(dur_counts.values())
            
            if total_dur > 0:
                small_pct = dur_counts[0] / total_dur * 100
                medium_pct = dur_counts[1] / total_dur * 100
                large_pct = dur_counts[2] / total_dur * 100
                print(f"{pair:<20} {small_pct:>8.2f}% {medium_pct:>8.2f}% {large_pct:>8.2f}%")
        
        total_all_dur = sum(total_dur_counts.values())
        if total_all_dur > 0:
            overall_small = total_dur_counts[0] / total_all_dur * 100
            overall_medium = total_dur_counts[1] / total_all_dur * 100
            overall_large = total_dur_counts[2] / total_all_dur * 100
            print(f"{'-'*60}")
            print(f"{'OVERALL':<20} {overall_small:>8.2f}% {overall_medium:>8.2f}% {overall_large:>8.2f}%")
        
        print(f"\n{'='*60}")
        print("✅ STATISTICS COMPLETE - Data examined and validated!")
        print(f"{'='*60}")
        
        # Create splits
        splits = self.create_splits(all_processed)
        
        # Save everything
        self.save_processed_data(splits)
        
        print(f"\n🎉 Pipeline complete! Data saved to {self.config.output_dir}")
        
        return splits


# ============================================================================
# Streaming Dataset for Causal Training
# ============================================================================

class StreamingCodeSwitchDataset(Dataset):
    """
    PyTorch Dataset for streaming code-switch prediction.
    Implements causal masking: at each position t, only sees tokens [0, t]
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        max_context_window: int = 128,
        return_causal_mask: bool = True,
        pad_token_id: int = 1
    ):
        """
        Args:
            data_path: Path to processed .pkl file
            max_context_window: Maximum number of previous tokens to consider
            return_causal_mask: Whether to return attention mask
        """
        with open(data_path, 'rb') as f:
            self.samples = pickle.load(f)
        
        self.max_context_window = max_context_window
        self.return_causal_mask = return_causal_mask
        self.pad_token_id = pad_token_id
        
        # Flatten samples into individual prediction points
        self.prediction_points = []
        for sample in self.samples:
            token_ids = sample['token_ids']
            switch_labels = sample['switch_labels']
            duration_labels = sample['duration_labels']
            
            for t in range(1, len(token_ids) - 1):  # Skip first and last
                # Only include valid prediction points
                if switch_labels[t] != -1:
                    self.prediction_points.append({
                        'sample_idx': len(self.prediction_points),
                        'token_ids': token_ids,
                        'position': t,
                        'switch_label': switch_labels[t],
                        'duration_label': duration_labels[t]
                    })
        
        print(f"Created dataset with {len(self.prediction_points)} prediction points")
    
    def __len__(self):
        return len(self.prediction_points)
    
    def __getitem__(self, idx):
        point = self.prediction_points[idx]
        
        t = point['position']
        token_ids = point['token_ids']
        
        # Extract causal context: tokens [max(0, t-window), t]
        start_idx = max(0, t - self.max_context_window + 1)
        context_ids = token_ids[start_idx:t+1]
        
        # Pad if necessary
        if len(context_ids) < self.max_context_window:
            padding = [self.pad_token_id] * (self.max_context_window - len(context_ids))
            context_ids = padding + context_ids
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1 if tid != self.pad_token_id else 0 for tid in context_ids]
        
        return {
            'input_ids': torch.tensor(context_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'switch_label': torch.tensor(point['switch_label'], dtype=torch.long),
            'duration_label': torch.tensor(point['duration_label'], dtype=torch.long)
        }


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Run the complete Phase 2 pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Process data for code-switching prediction")
    parser.add_argument(
        "--model_name",
        type=str,
        default="xlm-roberta-base",
        help="Tokenizer/backbone name (e.g. xlm-roberta-base or bert-base-multilingual-cased)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./processed_data",
        help="Directory to save processed splits",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./cache",
        help="Directory for cached artifacts",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Tokenizer max length metadata to store in config",
    )
    parser.add_argument(
        "--spanish_lid_backend",
        type=str,
        default="codeswitch",
        choices=["papluca", "codeswitch"],
        help="Backend used for Spanish-English word-level language identification",
    )

    args = parser.parse_args()

    # Initialize configuration
    config = ProcessingConfig(
        model_name=args.model_name,
        output_dir=Path(args.output_dir),
        cache_dir=Path(args.cache_dir),
        max_length=args.max_length,
        spanish_lid_backend=args.spanish_lid_backend,
    )
    
    # Create processor
    processor = SwitchLinguaProcessor(config)
    
    # Run pipeline
    splits = processor.run_complete_pipeline()
    
    # Test the streaming dataset
    print("\n" + "="*60)
    print("Testing Streaming Dataset")
    print("="*60)
    
    train_path = config.output_dir / "train.pkl"
    if train_path.exists():
        dataset = StreamingCodeSwitchDataset(
            train_path,
            max_context_window=64
        )
        
        # Test dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0
        )
        
        # Show first batch
        batch = next(iter(dataloader))
        print(f"\n📦 Sample batch:")
        print(f"   - input_ids shape: {batch['input_ids'].shape}")
        print(f"   - attention_mask shape: {batch['attention_mask'].shape}")
        print(f"   - switch_label shape: {batch['switch_label'].shape}")
        print(f"   - duration_label shape: {batch['duration_label'].shape}")
        print(f"   - Switch label distribution: {torch.bincount(batch['switch_label'][batch['switch_label'] != -1])}")
    
    print("\n✅ Phase 2 Complete!")


if __name__ == "__main__":
    main()
