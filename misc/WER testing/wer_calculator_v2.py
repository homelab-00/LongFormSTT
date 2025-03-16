from jiwer import wer, process_words, transforms
import os

# Text cleaning pipeline
text_leaner = transforms.Compose([
    transforms.RemovePunctuation(),
    transforms.ToLowerCase(),
    transforms.Strip(),
    transforms.RemoveMultipleSpaces()
])

def load_and_clean(file_path):
    """Load and clean text from a .txt file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_text = f.read().strip()
        return text_leaner(raw_text)
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} not found!")
        exit(1)

# Get absolute path of current directory
current_dir = os.path.abspath(os.path.dirname(__file__)) if "__file__" in locals() else os.getcwd()
print(f"🔍 Looking for files in: {current_dir}")

# Verify control.txt exists
control_path = "control.txt"
if not os.path.exists(control_path):
    print(f"❌ Critical Error: control.txt not found in directory!\n"
          f"   Please ensure control.txt exists here: {current_dir}")
    exit(1)

control_text = load_and_clean(control_path)

# Get comparison files
all_files = [f for f in os.listdir() if f.endswith('.txt') and f != control_path]
if not all_files:
    print("❌ No other .txt files found for comparison")
    exit(1)

print(f"\n✅ Found control.txt and {len(all_files)} files to compare:")

for file in all_files:
    print(f" - {file}")

print("\n🏁 Starting analysis...")

# Analysis loop
for hypothesis_file in all_files:
    try:
        hypothesis_text = load_and_clean(hypothesis_file)
        alignment = process_words(control_text, hypothesis_text)
        
        error_stats = {
            'wer': wer(control_text, hypothesis_text),
            'insertions': alignment.insertions,
            'deletions': alignment.deletions,
            'substitutions': alignment.substitutions,
            'total_words': len(alignment.truth_words)
        }
        
        print(f"\n📊 Results for {hypothesis_file}:")
        print(f"  Word Error Rate (WER): {error_stats['wer'] * 100:.2f}%")
        print(f"  Insertions: {error_stats['insertions']}")
        print(f"  Deletions: {error_stats['deletions']}")
        print(f"  Substitutions: {error_stats['substitutions']}")
        print(f"  Total Reference Words: {error_stats['total_words']}")
        
    except Exception as e:
        print(f"⚠️  Error processing {hypothesis_file}: {str(e)}")
        continue

print("\n✅ Analysis complete!")