import re

def extract_number(text):

    text = str(text)

    match = re.search(

        r'(-?\d+(?:\.\d+)?)',

        text
    )

    if match:
        return match.group(1)

    return None


def normalize_text(text):

    return str(text).strip().lower()


# MGSM

def evaluate_mgsm(correct_answer, response):

    gt = extract_number(correct_answer)

    pred = extract_number(response)

    return gt == pred


# IMDB

def evaluate_imdb(correct_answer, response):

    gt = "positive" if correct_answer == 1 else "negative"

    pred = normalize_text(response)

    return gt in pred


# MMLU

def evaluate_mmlu(correct_answer, response):

    gt = normalize_text(correct_answer)

    pred = normalize_text(response)

    return gt in pred