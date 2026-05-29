import re

# =====================================================
# NUMBER EXTRACTION
# =====================================================

def extract_number(text):

    text = str(text)

    # Final Answer 우선

    match = re.search(
        r'Final\s*Answer\s*:?\s*(-?\d+(?:\.\d+)?)',
        text,
        re.IGNORECASE
    )

    if match:
        return match.group(1)

    # 모든 숫자 찾기

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

    gt = (
        "positive"
        if correct_answer == 1
        else "negative"
    )

    pred = normalize_text(response)

    # 정확히 positive/negative만 판단

    if "positive" in pred:
        pred_label = "positive"

    elif "negative" in pred:
        pred_label = "negative"

    else:
        return False

    return gt == pred_label


# =====================================================
# MMLU
# =====================================================

def evaluate_mmlu(correct_answer, response):

    gt = normalize_text(correct_answer)

    pred = normalize_text(response)

    # 완전 일치 우선

    if pred == gt:
        return True

    # The answer is XXX

    if gt in pred:

        # 너무 짧은 정답 방지
        # ex) gt="4", pred="24"

        if len(gt) >= 3:
            return True

        words = re.findall(
            r'\b\w+\b',
            pred
        )

        return gt in words

    return False