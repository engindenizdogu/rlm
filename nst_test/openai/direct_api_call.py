import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Read the massive context file
with open("massive_context.txt", "r") as f:
    context = f.read()

# Create the prompt
prompt = "I'm looking for a magic number. I'm not sure if it's in this chunk, but tell me if you can find it."

# Make API call to OpenAI
response = client.chat.completions.create(
    model="gpt-5-nano-2025-08-07",
    messages=[
        {"role": "user", "content": f"{prompt}\n\nContent:\n{context}"}
    ]
)

# Print the response
print(response.choices[0].message.content)
