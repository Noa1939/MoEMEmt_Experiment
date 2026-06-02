import ollama

r = ollama.generate(
    model="gemma4:e4b",
    prompt="Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q. Answer only with a number.",
    options={
        "temperature": 0,
        "num_predict": 4096
    }
)

print(r.done_reason)
print(repr(r.response))