from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1234/v1/", api_key="lm-studio")

completion = client.chat.completions.create(
    model="phi4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's up with the pyramids?"},
    ]
)

print(completion.choices[0].message.content)
