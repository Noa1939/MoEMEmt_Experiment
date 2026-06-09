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

# MODELS = [
#     "gemma4:e4b"
# ]

MODELS = [

    "qwen2.5:14b",

    "gemma2:9b",

    "codellama:13b",
       
    "gemma3:12b",

    "gemma4:e4b"
]

# =====================================================
# METHODS
# =====================================================

METHODS = [
    "plain",
    "base64",
    "guided_base64",
    "caesar",
    "moe"
]


# METHODS = [
#     "plain",
#     "base64",
#     "caesar",
#     "moe"
# ]


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

GUIDED_BASE64_PROMPT = """
This text is encoded using Base64.

{guideline}

Decode the text and answer the task.

Return only the answer.

Encoded Text:
"""

# =====================================================
# TASK_GUIDELINES
# =====================================================

TASK_GUIDELINES = {

    "MGSM": """
Carefully reconstruct all numerical values.

Preserve quantities exactly.

Verify arithmetic relationships before solving.
""",

    "IMDB": """
Focus on emotional expressions.

Identify sentiment-bearing words.

Determine overall sentiment.
""",

    "MMLU": """
Identify the key concept.

Focus on factual correctness.

Choose the most likely answer.
"""
}


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
                    correct_answer = choices[answer_idx]

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

                # ===================================
                # METHOD : PLAIN
                # ===================================

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

                    # print("PLAIN =", repr(final_response))

                # ===================================
                # METHOD : BASE64
                # ===================================

                elif method == "base64":

                    encoded = encode_base64(
                        task_text
                    )

                    prompt = f"""
{BASE64_PROMPT}

{encoded}
"""

                    final_response = ask_model(
                        prompt,
                        MODEL_NAME
                    )

                    R1 = ""
                    R2 = final_response
                    R3 = ""

                # ===================================
                # METHOD : GUIDED_BASE64
                # ===================================

                elif method == "guided_base64":

                    encoded = encode_base64(
                        task_text
                    )

                    guideline = TASK_GUIDELINES[
                        dataset_name
                    ]

                    prompt = GUIDED_BASE64_PROMPT.format(
                        guideline=guideline
                    )

                    final_response = ask_model(
                        f"""
                {prompt}

                {encoded}
                    """,
                        MODEL_NAME
                    )

                    R1 = ""
                    R2 = final_response
                    R3 = ""

                # ===================================
                # METHOD : CAESAR
                # ===================================

                elif method == "caesar":

                    encoded = caesar_cipher(
                        task_text
                    )

                    prompt = f"""
{CAESAR_PROMPT}

{encoded}
"""

                    final_response = ask_model(
                        prompt,
                        MODEL_NAME
                    )

                    R1 = ""
                    R2 = ""
                    R3 = final_response

                # ===================================
                # METHOD : MOE
                # ===================================

                elif method == "moe":

                    R1 = ask_model(
                        f"""
{PLAIN_PROMPT}

{task_text}
""",
                        MODEL_NAME
                    )

                    encoded_b64 = encode_base64(
                        task_text
                    )

                    R2 = ask_model(
                        f"""
{BASE64_PROMPT}

{encoded_b64}
""",
                        MODEL_NAME
                    )

                    encoded_caesar = caesar_cipher(
                        task_text
                    )

                    R3 = ask_model(
                        f"""
{CAESAR_PROMPT}

{encoded_caesar}
""",
                        MODEL_NAME
                    )

                    # print("\n===== MOE DEBUG =====")
                    # print("R1 =", repr(R1))
                    # print("R2 =", repr(R2))
                    # print("R3 =", repr(R3))
                    # print("=====================\n")

                    moe_prompt = MOE_PROMPT.format(
                        A=R1,
                        B=R2,
                        C=R3
                    )

                    final_response = ask_model(
                        moe_prompt,
                        MODEL_NAME
                    )
                    
                    

                # ===================================
                # EVALUATION
                # ===================================

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

                # ===================================
                # SAVE ROW
                # ===================================

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

            # =======================================
            # SAVE CSV
            # =======================================

            df = pd.DataFrame(results)

            save_path = (
                f"{model_dir}/{method}_{dataset_name}.csv"
            )

            df.to_csv(
                save_path,
                index=False
            )

            accuracy = (
                df["correct"].mean() * 100
            )

            dataset_scores[method] = accuracy

            print(
                f"{dataset_name} {method}: {accuracy:.2f}"
            )

        # ===========================================
        # SUMMARY SAVE
        # ===========================================

        for method in METHODS:

            summary_results.append({

                "Model": MODEL_NAME,

                "Method": method,

                "Dataset": dataset_name,

                "Accuracy":
                    dataset_scores[method]
            })

# =====================================================
# FINAL SUMMARY
# =====================================================

summary_df = pd.DataFrame(
    summary_results
)

summary_df.to_csv(
    "outputs/summary.csv",
    index=False
)

paper_table = summary_df.pivot_table(
    index=["Model", "Method"],
    columns="Dataset",
    values="Accuracy"
)

paper_table.to_csv(
    "outputs/final_table.csv"
)

print("\n====================")
print("Experiment Finished")
print("====================")

print("\nSaved:")
print("outputs/summary.csv")
print("outputs/final_table.csv")

print(paper_table)

