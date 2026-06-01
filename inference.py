import ollama

def ask_model(prompt, model_name):

    response = ollama.generate(
        model=model_name,
        prompt=prompt,
        options={
            "temperature": 0,
            "num_predict": 512
        }
    )

    # print("MODEL:", model_name)
    # print("RAW:", response)
    # print("ANSWER:", repr(response.response))

    return response.response