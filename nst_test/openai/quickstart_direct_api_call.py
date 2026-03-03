import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

start = time.time()
response = client.chat.completions.create(
    model="gpt-5-nano-2025-08-07",
    messages=[
        {"role": "user", "content": "Print me the first 100 powers of two, each on a newline."}
    ]
)
result = response.choices[0].message.content
end = time.time()

print(result)
print(f"Time taken: {end - start} seconds")
