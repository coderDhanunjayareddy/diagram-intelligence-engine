import os
import openai

print("Checking OpenAI API...")
try:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello, write one word."}],
            max_tokens=5
        )
        print("OpenAI response:", response.choices[0].message.content.strip())
    else:
        print("OPENAI_API_KEY is not set.")
except Exception as e:
    print("OpenAI test failed:", e)
