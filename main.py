# main.py

from datasets import load_dataset
from inference import *
from evaluation import *
from encoding import *

import pandas as pd
from tqdm import tqdm
import os

# =========================================================
# MODELS
# =========================================================

MODELS = [

    "qwen2.5:14b",

    "gemma2:9b",

    "codellama:13b"
]

# =========================================================
# METHODS
# =========================================================

METHODS = [

    "plain",

    "base64",

    "caesar",

    "moe"
]

# =========================================================
# OUTPUT
# =========================================================

os.makedirs("outputs", exist_ok=True)

# =========================================================
# PROMPTS
# =========================================================

PLAIN_PROMPT = """
Solve the following task carefully.

Think step-by-step if needed.

At the end,
give the final answer clearly.
"""

# =========================================================
# BASE64 PROMPT
# =========================================================

BASE64_PROMPT = """
The following text is encoded using Base64 encoding.

Your task is:

1. Decode the Base64 text carefully.
2. Recover the original English sentence.
3. Solve the recovered task step-by-step.
4. Give the final answer clearly.

Do NOT explain Base64 itself.

If this is a math problem,
write at the end:

Final Answer: <number>

Encoded Text:
"""

# =========================================================
# CAESAR PROMPT
# =========================================================

CAESAR_PROMPT = """
The following sentence is encoded using a Caesar cipher.

The alphabet was shifted forward by 3 characters.

Your task is:

1. Decode the Caesar cipher carefully.
2. Recover the original English sentence.
3. Solve the recovered task step-by-step.
4. Give the final answer clearly.

Do NOT generate example cryptography sentences.

If this is a math problem,
write at the end:

Final Answer: <number>

Encoded Text:
"""

# =========================================================
# MOE PROMPT
# =========================================================

MOE_PROMPT = """
You are given answers from three different reasoning systems.

A is usually the most reliable.

B and C were generated from encoded versions of the same task.

Use B and C only if they semantically agree with A.

Ignore unrelated or cryptography-example outputs.

Return only the final answer.

A:
{A}

B:
{B}

C:
{C}
"""

# =========================================================
# LOAD DATASETS
# =========================================================

print("Loading datasets...")

# ---------------------------------------------------------
# MMLU
# ---------------------------------------------------------

mmlu = load_dataset("cais/mmlu", "all")

mmlu_samples = mmlu["test"].select(range(20))

# ---------------------------------------------------------
# MGSM
# ---------------------------------------------------------

mgsm = load_dataset("juletxara/mgsm", "en")

mgsm_samples = mgsm["test"].select(range(20))

# ---------------------------------------------------------
# IMDB
# ---------------------------------------------------------

imdb = load_dataset("imdb")

imdb_samples = imdb["test"].select(range(20))

# =========================================================
# DATASET DICT
# =========================================================

datasets_dict = {

    "MMLU": mmlu_samples,

    "MGSM": mgsm_samples,

    "IMDB": imdb_samples
}

# =========================================================
# MODEL LOOP
# =========================================================

for MODEL_NAME in MODELS:

    print(f"\n======================")
    print(f"MODEL: {MODEL_NAME}")
    print(f"======================")

    safe_model_name = MODEL_NAME.replace(":", "_")

    model_dir = f"outputs/{safe_model_name}"

    os.makedirs(model_dir, exist_ok=True)

    # =====================================================
    # DATASET LOOP
    # =====================================================

    for dataset_name, samples in datasets_dict.items():

        print(f"\n======================")
        print(f"DATASET: {dataset_name}")
        print(f"======================")

        # =================================================
        # METHOD LOOP
        # =================================================

        for method in METHODS:

            print(f"\nRunning method: {method}")

            results = []

            # =============================================
            # SAMPLE LOOP
            # =============================================

            for sample in tqdm(samples):

                # =================================================
                # DATASET PARSING
                # =================================================

                # -------------------------------------------------
                # MMLU
                # -------------------------------------------------

                if dataset_name == "MMLU":

                    question = sample["question"]

                    choices = sample["choices"]

                    answer_idx = sample["answer"]

                    correct_answer = choices[answer_idx]

                    task_text = f"""
                    Question:
                    {question}

                    Choices:
                    {choices}

                    Only answer with the correct choice.
                    """

                # -------------------------------------------------
                # MGSM
                # -------------------------------------------------

                elif dataset_name == "MGSM":

                    question = sample["question"]

                    correct_answer = sample["answer_number"]

                    task_text = f"""
                    Solve this math problem carefully.

                    At the end write exactly:

                    Final Answer: <number>

                    Question:
                    {question}
                    """

                # -------------------------------------------------
                # IMDB
                # -------------------------------------------------

                elif dataset_name == "IMDB":

                    question = sample["text"]

                    correct_answer = sample["label"]

                    task_text = f"""
                    Review:
                    {question}

                    Is this review positive or negative?

                    Answer only:
                    positive
                    or
                    negative
                    """

                # =================================================
                # METHOD 1 : PLAIN
                # =================================================

                if method == "plain":

                    final_response = ask_model(

                        f"""
                        {PLAIN_PROMPT}

                        {task_text}
                        """,

                        MODEL_NAME
                    )

                    R1 = final_response
                    R2 = ""
                    R3 = ""

                # =================================================
                # METHOD 2 : BASE64
                # =================================================

                elif method == "base64":

                    encoded = encode_base64(task_text)

                    base64_input = f"""
                    {BASE64_PROMPT}

                    {encoded}
                    """

                    final_response = ask_model(

                        base64_input,

                        MODEL_NAME
                    )

                    R1 = ""
                    R2 = final_response
                    R3 = ""

                # =================================================
                # METHOD 3 : CAESAR
                # =================================================

                elif method == "caesar":

                    encoded = caesar_cipher(task_text)

                    caesar_input = f"""
                    {CAESAR_PROMPT}

                    {encoded}
                    """

                    final_response = ask_model(

                        caesar_input,

                        MODEL_NAME
                    )

                    R1 = ""
                    R2 = ""
                    R3 = final_response

                # =================================================
                # METHOD 4 : MOE
                # =================================================

                elif method == "moe":

                    # ------------------------------------------------
                    # R1
                    # ------------------------------------------------

                    R1 = ask_model(

                        f"""
                        {PLAIN_PROMPT}

                        {task_text}
                        """,

                        MODEL_NAME
                    )

                    # ------------------------------------------------
                    # R2
                    # ------------------------------------------------

                    encoded_b64 = encode_base64(task_text)

                    base64_input = f"""
                    {BASE64_PROMPT}

                    {encoded_b64}
                    """

                    R2 = ask_model(

                        base64_input,

                        MODEL_NAME
                    )

                    # ------------------------------------------------
                    # R3
                    # ------------------------------------------------

                    encoded_caesar = caesar_cipher(task_text)

                    caesar_input = f"""
                    {CAESAR_PROMPT}

                    {encoded_caesar}
                    """

                    R3 = ask_model(

                        caesar_input,

                        MODEL_NAME
                    )

                    # ------------------------------------------------
                    # AGGREGATION
                    # ------------------------------------------------

                    moe_input = MOE_PROMPT.format(

                        A=R1,
                        B=R2,
                        C=R3
                    )

                    final_response = ask_model(

                        moe_input,

                        MODEL_NAME
                    )

                # =================================================
                # EVALUATION
                # =================================================

                if dataset_name == "MGSM":

                    correct = evaluate_mgsm(

                        correct_answer,

                        final_response
                    )

                elif dataset_name == "IMDB":

                    correct = evaluate_imdb(

                        correct_answer,

                        final_response
                    )

                else:

                    correct = evaluate_mmlu(

                        correct_answer,

                        final_response
                    )

                # =================================================
                # SAVE RESULT
                # =================================================

                results.append({

                    "question": question,

                    "correct_answer": correct_answer,

                    "method": method,

                    "R1": R1,
                    "R2": R2,
                    "R3": R3,

                    "FINAL": final_response,

                    "correct": correct
                })

            # =====================================================
            # SAVE CSV
            # =====================================================

            df = pd.DataFrame(results)

            save_path = f"{model_dir}/{method}_{dataset_name}.csv"

            df.to_csv(save_path, index=False)

            accuracy = df["correct"].mean() * 100

            print(f"\n{dataset_name} Accuracy ({method}): {accuracy:.2f}")

            print(f"Saved to: {save_path}")