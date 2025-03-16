#!/usr/bin/env python3
import os
import re
import logging

# -----------------------------------------------------------
# Configuration for logging (Debug level ~7/10)
# -----------------------------------------------------------
logging.basicConfig(level=logging.INFO,  # Switch to INFO if you want fewer messages
                    format='%(levelname)s: %(message)s')

def clean_text(text: str) -> str:
    """
    Remove punctuation, special symbols, and normalize case from the input text.
    Returns the cleaned text.
    """
    logging.debug("Original text for cleaning: %s", text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation & other non-word characters
    text = re.sub(r"[^\w\s]", "", text)
    # Strip extra whitespace
    text = text.strip()
    logging.debug("Cleaned text: %s", text)
    return text

def wer_calculation(reference_words, hypothesis_words):
    """
    Calculate Word Error Rate using a dynamic programming approach.
    Returns the tuple (S, I, D, WER).
    - reference_words: list of words from the reference (control.txt)
    - hypothesis_words: list of words from the hypothesis (transcription under test)
    """
    # We'll use the standard approach to compute Levenshtein distance
    # where cost(S, I, D) is used to derive WER.

    # Create a 2D matrix (dp) with size (len(reference_words)+1) x (len(hypothesis_words)+1)
    # dp[i][j] will hold the edit distance (S + I + D) between the first i words
    # of reference and first j words of hypothesis.
    rows = len(reference_words) + 1
    cols = len(hypothesis_words) + 1
    dp = [[0] * cols for _ in range(rows)]

    # Initialize the first column and the first row of the matrix
    for i in range(1, rows):
        dp[i][0] = i
    for j in range(1, cols):
        dp[0][j] = j

    # Fill in the dp matrix
    for i in range(1, rows):
        for j in range(1, cols):
            if reference_words[i-1] == hypothesis_words[j-1]:
                dp[i][j] = dp[i-1][j-1]  # no change
            else:
                substitution = dp[i-1][j-1] + 1
                insertion    = dp[i][j-1] + 1
                deletion     = dp[i-1][j] + 1
                dp[i][j] = min(substitution, insertion, deletion)

    # The edit distance is dp[rows-1][cols-1]
    edit_distance = dp[rows-1][cols-1]
    # The total number of reference words
    ref_len = len(reference_words)

    # We need to backtrack to find the actual S, I, D counts
    i, j = rows - 1, cols - 1
    S = I = D = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference_words[i-1] == hypothesis_words[j-1]:
            # Same word, no error
            i -= 1
            j -= 1
        else:
            # Identify the minimum of the possible moves
            current_val = dp[i][j]
            if i > 0 and j > 0 and dp[i-1][j-1] + 1 == current_val:
                S += 1
                i -= 1
                j -= 1
            elif j > 0 and dp[i][j-1] + 1 == current_val:
                I += 1
                j -= 1
            elif i > 0 and dp[i-1][j] + 1 == current_val:
                D += 1
                i -= 1

    # WER is (S + I + D) / number_of_words_in_reference
    wer_value = float(edit_distance) / ref_len if ref_len > 0 else 0.0

    return S, I, D, wer_value

def main():
    logging.info("Starting comparison script...")

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logging.debug(f"Script directory: {script_dir}")

    # Identify all .txt files in the script's directory
    files_in_dir = [f for f in os.listdir(script_dir) if f.endswith(".txt")]
    logging.debug(f"Text files found: {files_in_dir}")

    # Ensure that 'control.txt' exists
    if 'control.txt' not in files_in_dir:
        logging.error("No 'control.txt' file found in the script's directory!")
        return

    # Load control transcription
    logging.debug("Loading control.txt reference...")
    with open(os.path.join(script_dir, 'control.txt'), 'r', encoding='utf-8') as f:
        control_text = clean_text(f.read())
    control_words = control_text.split()

    # Remove 'control.txt' from the list so we only compare the others
    files_in_dir.remove('control.txt')

    if not files_in_dir:
        logging.warning("No other transcription files found for comparison.")
        return

    # Compare each transcription file with control.txt
    for txt_file in files_in_dir:
        logging.info(f"Comparing '{txt_file}' against control.txt")
        with open(os.path.join(script_dir, txt_file), 'r', encoding='utf-8') as f:
            hypothesis_text = clean_text(f.read())
        hypothesis_words = hypothesis_text.split()

        # Calculate the WER
        S, I, D, wer_value = wer_calculation(control_words, hypothesis_words)

        # Print / log results
        logging.info(
            f"File: {txt_file}\n"
            f"  Substitutions (S): {S}\n"
            f"  Insertions (I):    {I}\n"
            f"  Deletions (D):     {D}\n"
            f"  WER:               {wer_value:.3f}\n"
            "-----------------------------------"
        )

if __name__ == "__main__":
    main()
