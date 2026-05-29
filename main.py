from datasets import load_dataset
from inference import ask_model
from evaluation import *
from encoding import *

import pandas as pd
from tqdm import tqdm
import os

# =====================================================
# MODELS
# =====================================================

MODELS = [

    "qwen2.5:14b",

    "gemma2:9b",

    "codellama:13b"
]

# =====================================================
# METHODS
# =====================================================

METHODS = [

    "plain",

    "base64",

    "caesar",

    "moe"
]

# =====================================================
# OUTPUT
# =====================================================

os.makedirs("outputs", exist_ok=True)

# =====================================================
# PROMPTS
# =====================================================

PLAIN_PROMPT = """
Solve the task.

Return only the final answer.
"""

BASE64_PROMPT = """
This text is encoded using Base64.

Decode it and answer the task.

Return only the answer.

Encoded Text:
"""

CAESAR_PROMPT = """
This text is encoded using a Caesar cipher (shift=3).

Decode it and answer the task.

Return only the answer.

Encoded Text:
"""

MOE_PROMPT = """
You are given answers from three systems.

A:
{A}

B:
{B}

C:
{C}

Return only the best final answer.
"""

# =====================================================
# LOAD DATASETS
# =====================================================

print("Loading datasets...")

mmlu = load_dataset(
    "cais/mmlu",
    "all"
)

mgsm = load_dataset(
    "juletxara/mgsm",
    "en"
)

imdb = load_dataset(
    "imdb"
)

# =====================================================
# SAMPLE SIZE
# =====================================================

SAMPLE_SIZE = 20

mmlu_samples = mmlu["test"].select(
    range(SAMPLE_SIZE)
)

mgsm_samples = mgsm["test"].select(
    range(SAMPLE_SIZE)
)

imdb_samples = imdb["test"].select(
    range(SAMPLE_SIZE)
)

datasets_dict = {

    "MMLU": mmlu_samples,

    "MGSM": mgsm_samples,

    "IMDB": imdb_samples
}

# =====================================================
# SUMMARY
# =====================================================

summary_results = []

# =====================================================
# MODEL LOOP
# =====================================================

for MODEL_NAME in MODELS:

    print("\n====================")
    print(MODEL_NAME)
    print("====================")

    safe_model_name = MODEL_NAME.replace(":", "_")

    model_dir = f"outputs/{safe_model_name}"

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    # ===============================================
    # DATASET LOOP
    # ===============================================

    for dataset_name, samples in datasets_dict.items():

        print(f"\nDataset: {dataset_name}")

        dataset_scores = {}

        # ===========================================
        # METHOD LOOP
        # ===========================================

        for method in METHODS:

            print(f"\nMethod: {method}")

            results = []

            for sample in tqdm(samples):

                # ===================================
                # MMLU
                # ===================================

                if dataset_name == "MMLU":

                    question = sample["question"]

                    choices = sample["choices"]

                    answer_idx = sample["answer"]

                    correct_answer = choices[
                        answer_idx
                    ]

                    task_text = f"""
Question:
{question}

Choices:
{choices}

Return only the correct answer.
"""

                # ===================================
                # MGSM
                # ===================================

                elif dataset_name == "MGSM":

                    question = sample["question"]

                    correct_answer = sample[
                        "answer_number"
                    ]

                    task_text = f"""
Solve the math problem.

Question:
{question}

Return only the final number.
"""

                # ===================================
                # IMDB
                # ===================================

                elif dataset_name == "IMDB":

                    question = sample["text"]

                    correct_answer = sample[
                        "label"
                    ]

                    task_text = f"""
Review:
{question}

Return only:

positive

or

negative
"""
from datasets import load_dataset
from inference import ask_model
from evaluation import *
from encoding import *

import pandas as pd
from tqdm import tqdm
import os

# =====================================================
# MODELS
# =====================================================

MODELS = [

    "qwen2.5:14b",

    "gemma2:9b",

    "codellama:13b"
]

# =====================================================
# METHODS
# =====================================================

METHODS = [

    "plain",

    "base64",

    "caesar",

    "moe"
]

# =====================================================
# OUTPUT
# =====================================================

os.makedirs("outputs", exist_ok=True)

# =====================================================
# PROMPTS
# =====================================================

PLAIN_PROMPT = """
Solve the task.

Return only the final answer.
"""

BASE64_PROMPT = """
This text is encoded using Base64.

Decode it and answer the task.

Return only the answer.

Encoded Text:
"""

CAESAR_PROMPT = """
This text is encoded using a Caesar cipher (shift=3).

Decode it and answer the task.

Return only the answer.

Encoded Text:
"""

MOE_PROMPT = """
You are given answers from three systems.

A:
{A}

B:
{B}

C:
{C}

Return only the best final answer.
"""

# =====================================================
# LOAD DATASETS
# =====================================================

print("Loading datasets...")

mmlu = load_dataset(
    "cais/mmlu",
    "all"
)

mgsm = load_dataset(
    "juletxara/mgsm",
    "en"
)

imdb = load_dataset(
    "imdb"
)

# =====================================================
# SAMPLE SIZE
# =====================================================

SAMPLE_SIZE = 20

mmlu_samples = mmlu["test"].select(
    range(SAMPLE_SIZE)
)

mgsm_samples = mgsm["test"].select(
    range(SAMPLE_SIZE)
)

imdb_samples = imdb["test"].select(
    range(SAMPLE_SIZE)
)

datasets_dict = {

    "MMLU": mmlu_samples,

    "MGSM": mgsm_samples,

    "IMDB": imdb_samples
}

# =====================================================
# SUMMARY
# =====================================================

summary_results = []

# =====================================================
# MODEL LOOP
# =====================================================

for MODEL_NAME in MODELS:

    print("\n====================")
    print(MODEL_NAME)
    print("====================")

    safe_model_name = MODEL_NAME.replace(":", "_")

    model_dir = f"outputs/{safe_model_name}"

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    # ===============================================
    # DATASET LOOP
    # ===============================================

    for dataset_name, samples in datasets_dict.items():

        print(f"\nDataset: {dataset_name}")

        dataset_scores = {}

        # ===========================================
        # METHOD LOOP
        # ===========================================

        for method in METHODS:

            print(f"\nMethod: {method}")

            results = []

            for sample in tqdm(samples):

                # ===================================
                # MMLU
                # ===================================

                if dataset_name == "MMLU":

                    question = sample["question"]

                    choices = sample["choices"]

                    answer_idx = sample["answer"]

                    correct_answer = choices[
                        answer_idx
                    ]

                    task_text = f"""
Question:
{question}

Choices:
{choices}

Return only the correct answer.
"""

                # ===================================
                # MGSM
                # ===================================

                elif dataset_name == "MGSM":

                    question = sample["question"]

                    correct_answer = sample[
                        "answer_number"
                    ]

                    task_text = f"""
Solve the math problem.

Question:
{question}

Return only the final number.
"""

                # ===================================
                # IMDB
                # ===================================

                elif dataset_name == "IMDB":

                    question = sample["text"]

                    correct_answer = sample[
                        "label"
                    ]

                    task_text = f"""
Review:
{question}

Return only:

positive

or

negative
"""