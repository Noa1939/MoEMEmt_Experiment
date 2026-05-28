# evaluation.py

import re

# =====================================================
# NUMBER EXTRACTION
# =====================================================

def extract_number(text):

    text = str(text)

    match = re.search(

        r'Final Answer:\s*(-?\d+(?:\.\d+)?)',

        text,

        re.IGNORECASE
    )

    if match:
        return match.group(1)

    matches = re.findall(

        r'-?\d+(?:\.\d+)?',

        text
    )

    if matches:
        return matches[-1]

    return None

# =====================================================
# NORMALIZATION
# =====================================================

def normalize_text(text):

    return str(text).strip().lower()

# =====================================================
# MGSM
# =====================================================

def evaluate_mgsm(correct_answer, response):

    gt = extract_number(correct_answer)

    pred = extract_number(response)

    return gt == pred

# =====================================================
# IMDB
# =====================================================

def evaluate_imdb(correct_answer, response):

    if correct_answer == 0:
        gt = "negative"
    else:
        gt = "positive"

    pred = normalize_text(response)

    return gt in pred

# =====================================================
# MMLU
# =====================================================

def evaluate_mmlu(correct_answer, response):

    gt = normalize_text(correct_answer)

    pred = normalize_text(response)

    return gt in pred