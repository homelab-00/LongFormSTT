from jiwer import process_words, transforms
import os

# Text cleaning pipeline
text_cleaner = transforms.Compose([
    transforms.RemovePunctuation(),
    transforms.ToLowerCase(),
    transforms.Strip(),
    transforms.RemoveMultipleSpaces()
])

def load_and_clean(file_path):
    """Load and clean text using script's directory as base"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(script_dir, file_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            raw_text = f.read().strip()
        return text_cleaner(raw_text)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        exit(1)

# Verify control.txt exists
control_path = "control.txt"
script_dir = os.path.dirname(os.path.abspath(__file__))
control_full_path = os.path.join(script_dir, control_path)

if not os.path.exists(control_full_path):
    print(f"❌ Missing control.txt in: {script_dir}")
    exit(1)

# Load control text
control_text = load_and_clean(control_path)

# Get all other .txt files
all_files = [f for f in os.listdir(script_dir) 
             if f.endswith('.txt') and f != control_path]

print(f"\n✅ Found control.txt and {len(all_files)} files to compare:")
for file in all_files:
    print(f" - {file}")

print("\n🏁 Starting analysis...")

for hypothesis_file in all_files:
    try:
        hypothesis_text = load_and_clean(hypothesis_file)
        
        # Process with alignment
        alignment = process_words(control_text, hypothesis_text)
        
        error_stats = {
            'wer': alignment.wer,
            'insertions': alignment.insertions,
            'deletions': alignment.deletions,
            'substitutions': alignment.substitutions,
            'total_words': sum(len(words) for words in alignment.references)  # Updated line
        }
        
        print(f"\n📊 Results for {hypothesis_file}:")
        print(f"  WER: {error_stats['wer'] * 100:.2f}%")
        print(f"  Insertions: {error_stats['insertions']}")
        print(f"  Deletions: {error_stats['deletions']}")
        print(f"  Substitutions: {error_stats['substitutions']}")
        print(f"  Total Reference Words: {error_stats['total_words']}")
        
    except Exception as e:
        print(f"⚠️  Error processing {hypothesis_file}: {str(e)}")
        continue

print("\n✅ Analysis complete!")