from jiwer import wer
import os

def load_text(file_path):
    """Φορτώνει το κείμενο από ένα .txt αρχείο."""
    script_dir = os.path.dirname(__file__)
    full_path = os.path.join(script_dir, file_path)
    with open(full_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

# Φόρτωση των αρχείων
control_text = load_text("control.txt")
hypothesis1_text = load_text("output1.txt")
hypothesis2_text = load_text("output2.txt")

# Υπολογισμός WER για κάθε υπόθεση
wer1 = wer(control_text, hypothesis1_text)
wer2 = wer(control_text, hypothesis2_text)

# Εκτύπωση αποτελεσμάτων
print(f"WER για output1.txt: {wer1 * 100:.2f}%")
print(f"WER για output2.txt: {wer2 * 100:.2f}%")