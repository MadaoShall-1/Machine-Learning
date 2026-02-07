"""
Streaming Simulation Demo for Code-Switching Prediction
========================================================
A demonstration script that processes text word-by-word and outputs:
1. Next-Token Switch Probability
2. Anticipated Duration (Small/Medium/Large)

This is one of the final deliverables for the CS 5100/7100 project.

Usage:
    python streaming_demo.py --model_path ./checkpoints/best_model.pt
    python streaming_demo.py --interactive  # For interactive mode

Author: Shuhuan Ye, Zhihang Cheng, Qi Zhou
Date: February 2026
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import json
import time
import sys

# For colorful output
try:
    from colorama import init, Fore, Back, Style
    init()
    HAS_COLORS = True
except ImportError:
    HAS_COLORS = False
    class Fore:
        RED = GREEN = YELLOW = BLUE = CYAN = MAGENTA = WHITE = RESET = ''
    class Back:
        RED = GREEN = YELLOW = BLUE = RESET = ''
    class Style:
        BRIGHT = RESET_ALL = ''


class StreamingCodeSwitchDemo:
    """
    Interactive demonstration of streaming code-switching prediction.
    
    Processes text incrementally (word-by-word or token-by-token) and
    visualizes the model's predictions in real-time.
    """
    
    # Duration class descriptions
    DURATION_CLASSES = {
        0: ('Small', '1-2 tokens', 'Lexical insertion'),
        1: ('Medium', '3-6 tokens', 'Phrase-level'),
        2: ('Large', '7+ tokens', 'Clausal/Sentential')
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = 'xlm-roberta-base',
        device: str = 'auto'
    ):
        """
        Initialize the demo.
        
        Args:
            model_path: Path to trained model checkpoint (optional)
            model_name: Base model name for tokenizer
            device: 'cuda', 'cpu', or 'auto'
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"{Fore.CYAN}🚀 Initializing Streaming Demo{Style.RESET_ALL}")
        print(f"   Device: {self.device}")
        
        # Load tokenizer
        print(f"   Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model
        if model_path and Path(model_path).exists():
            print(f"   Loading trained model from: {model_path}")
            self._load_trained_model(model_path)
        else:
            print(f"   {Fore.YELLOW}⚠️ No trained model provided - using random weights for demo{Style.RESET_ALL}")
            self._create_demo_model()
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"   {Fore.GREEN}✅ Ready!{Style.RESET_ALL}\n")
    
    def _load_trained_model(self, model_path: str):
        """Load a trained model from checkpoint"""
        from model_architecture_phase2 import CausalCodeSwitchModel, ModelConfig
        
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint.get('config', ModelConfig())
        
        self.model = CausalCodeSwitchModel(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def _create_demo_model(self):
        """Create a model with random weights for demonstration"""
        from model_architecture_phase2 import CausalCodeSwitchModel, ModelConfig
        
        config = ModelConfig(
            model_name='xlm-roberta-base',
            max_context_window=128
        )
        self.model = CausalCodeSwitchModel(config)
    
    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        """
        Get predictions for the current text context.
        
        Args:
            text: Text processed so far
        
        Returns:
            dict with predictions and probabilities
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            add_special_tokens=True,
            truncation=True,
            max_length=128
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Forward pass with causal masking
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            apply_causal_mask=True
        )
        
        # Get predictions for the last position
        switch_logits = outputs['switch_logits'][0, -1, :]  # [2]
        duration_logits = outputs['duration_logits'][0, -1, :]  # [3]
        
        # Convert to probabilities
        switch_probs = F.softmax(switch_logits, dim=0)
        duration_probs = F.softmax(duration_logits, dim=0)
        
        # Get predictions
        p_switch = switch_probs[1].item()
        duration_class = torch.argmax(duration_probs).item()
        
        return {
            'p_switch': p_switch,
            'p_no_switch': switch_probs[0].item(),
            'predicted_switch': p_switch > 0.5,
            'duration_class': duration_class,
            'duration_probs': duration_probs.cpu().numpy(),
            'num_tokens': input_ids.shape[1]
        }
    
    def stream_text(
        self,
        text: str,
        delay: float = 0.3,
        word_by_word: bool = True
    ) -> List[Dict]:
        """
        Process text incrementally with visualization.
        
        Args:
            text: Full text to process
            delay: Delay between words (seconds)
            word_by_word: If True, process word by word; else token by token
        """
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}STREAMING CODE-SWITCH PREDICTION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
        print(f"Input text: {Fore.WHITE}{text}{Style.RESET_ALL}\n")
        print(f"{'='*70}\n")
        
        results = []
        
        if word_by_word:
            words = text.split()
            current_text = ""
            
            for i, word in enumerate(words):
                current_text = " ".join(words[:i+1])
                
                # Get prediction
                pred = self.predict(current_text)
                pred['position'] = i
                pred['word'] = word
                pred['context'] = current_text
                results.append(pred)
                
                # Visualize
                self._visualize_prediction(i, word, pred)
                
                # Delay for effect
                if delay > 0:
                    time.sleep(delay)
        
        else:
            # Token by token
            encoding = self.tokenizer(text, add_special_tokens=True)
            tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
            
            for i in range(1, len(tokens)):
                partial_ids = encoding['input_ids'][:i+1]
                partial_text = self.tokenizer.decode(partial_ids)
                
                pred = self.predict(partial_text)
                pred['position'] = i
                pred['token'] = tokens[i]
                pred['context'] = partial_text
                results.append(pred)
                
                self._visualize_prediction(i, tokens[i], pred, is_token=True)
                
                if delay > 0:
                    time.sleep(delay)
        
        # Summary
        self._print_summary(results)
        
        return results
    
    def _visualize_prediction(
        self,
        position: int,
        word: str,
        pred: Dict,
        is_token: bool = False
    ):
        """Visualize a single prediction"""
        p_switch = pred['p_switch']
        
        # Color based on switch probability
        if p_switch > 0.7:
            color = Fore.RED
            indicator = "🔴"
        elif p_switch > 0.5:
            color = Fore.YELLOW
            indicator = "🟡"
        elif p_switch > 0.3:
            color = Fore.BLUE
            indicator = "🔵"
        else:
            color = Fore.GREEN
            indicator = "🟢"
        
        # Progress bar for probability
        bar_len = 20
        filled = int(p_switch * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        
        # Duration info (only if switch predicted)
        duration_str = ""
        if pred['predicted_switch']:
            dur_class = pred['duration_class']
            dur_name, dur_range, dur_desc = self.DURATION_CLASSES[dur_class]
            duration_str = f" → {Fore.MAGENTA}Duration: {dur_name} ({dur_range}){Style.RESET_ALL}"
        
        # Print
        unit = "token" if is_token else "word"
        print(f"{indicator} [{position:3d}] {unit}: {color}{word:15s}{Style.RESET_ALL} "
              f"P(switch)=[{bar}] {p_switch:.1%}{duration_str}")
    
    def _print_summary(self, results: List[Dict]):
        """Print summary statistics"""
        print(f"\n{'='*70}")
        print(f"{Fore.CYAN}SUMMARY{Style.RESET_ALL}")
        print(f"{'='*70}")
        
        total = len(results)
        predicted_switches = sum(1 for r in results if r['predicted_switch'])
        avg_p_switch = sum(r['p_switch'] for r in results) / total if total > 0 else 0
        
        print(f"\n📊 Statistics:")
        print(f"   Total positions analyzed: {total}")
        print(f"   Predicted switches: {predicted_switches} ({predicted_switches/total:.1%})")
        print(f"   Average P(switch): {avg_p_switch:.2%}")
        
        # Duration distribution at predicted switches
        if predicted_switches > 0:
            duration_counts = {0: 0, 1: 0, 2: 0}
            for r in results:
                if r['predicted_switch']:
                    duration_counts[r['duration_class']] += 1
            
            print(f"\n📏 Duration Distribution (at predicted switches):")
            for cls, count in duration_counts.items():
                name, range_str, desc = self.DURATION_CLASSES[cls]
                pct = count / predicted_switches if predicted_switches > 0 else 0
                print(f"   {name} ({range_str}): {count} ({pct:.1%})")
        
        # High-probability switch positions
        high_prob_switches = [(r['position'], r['word'] if 'word' in r else r['token'], r['p_switch']) 
                             for r in results if r['p_switch'] > 0.5]
        
        if high_prob_switches:
            print(f"\n🎯 High-probability switch positions (P > 50%):")
            for pos, word, prob in high_prob_switches[:5]:
                print(f"   Position {pos}: '{word}' - {prob:.1%}")
    
    def interactive_mode(self):
        """Run in interactive mode"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}INTERACTIVE CODE-SWITCHING PREDICTION{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"\nType a code-switched sentence and press Enter to analyze.")
        print(f"Commands: 'quit' to exit, 'help' for examples\n")
        
        examples = [
            "Hola, how are you doing today?",
            "I went to the tienda to buy some milk.",
            "Yesterday fue un día muy interesting en la oficina.",
            "आज मैं office late पहुंचा because of traffic.",
            "这个 project 真的 very complicated.",
        ]
        
        while True:
            try:
                user_input = input(f"{Fore.GREEN}>>> {Style.RESET_ALL}").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye! 👋")
                    break
                
                elif user_input.lower() == 'help':
                    print(f"\n{Fore.CYAN}Example sentences:{Style.RESET_ALL}")
                    for i, ex in enumerate(examples, 1):
                        print(f"  {i}. {ex}")
                    print()
                
                elif user_input.isdigit() and 1 <= int(user_input) <= len(examples):
                    # Use example sentence
                    self.stream_text(examples[int(user_input) - 1], delay=0.2)
                
                elif len(user_input) > 0:
                    self.stream_text(user_input, delay=0.2)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")


def demo_predefined_examples():
    """Run demo with predefined code-switched examples"""
    
    demo = StreamingCodeSwitchDemo()
    
    examples = [
        # Spanish-English
        ("Spanish-English", "Ayer I went to the tienda and compré some bread."),
        
        # Hindi-English (Romanized)
        ("Hindi-English", "Main office mein bahut busy tha because of the deadline."),
        
        # Mixed with longer switches
        ("Long Switch", "The meeting was muy importante for the company strategy."),
    ]
    
    for lang_pair, text in examples:
        print(f"\n\n{Fore.MAGENTA}{'#'*70}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Example: {lang_pair}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}{'#'*70}{Style.RESET_ALL}")
        
        demo.stream_text(text, delay=0.15)
        
        input(f"\n{Fore.CYAN}Press Enter to continue to next example...{Style.RESET_ALL}")


def main():
    parser = argparse.ArgumentParser(
        description='Streaming Code-Switch Prediction Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python streaming_demo.py --interactive
  python streaming_demo.py --text "Hola, how are you today?"
  python streaming_demo.py --model_path ./checkpoints/best_model.pt --text "Ayer fue un busy day"
  python streaming_demo.py --demo
        """
    )
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--text', type=str, default=None,
                        help='Text to analyze')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--demo', action='store_true',
                        help='Run predefined demo examples')
    parser.add_argument('--delay', type=float, default=0.2,
                        help='Delay between words in seconds')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to run on')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_predefined_examples()
    elif args.interactive:
        demo = StreamingCodeSwitchDemo(
            model_path=args.model_path,
            device=args.device
        )
        demo.interactive_mode()
    elif args.text:
        demo = StreamingCodeSwitchDemo(
            model_path=args.model_path,
            device=args.device
        )
        demo.stream_text(args.text, delay=args.delay)
    else:
        # Default: run interactive mode
        demo = StreamingCodeSwitchDemo(
            model_path=args.model_path,
            device=args.device
        )
        demo.interactive_mode()


if __name__ == "__main__":
    main()
