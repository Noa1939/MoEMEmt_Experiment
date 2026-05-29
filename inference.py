import ollama

def ask_model(prompt, model_name):

    response = ollama.generate(

        model=model_name,

        prompt=prompt,

        options={

            "temperature": 0,

            "num_predict": 64
        }
    )

    return response["response"]