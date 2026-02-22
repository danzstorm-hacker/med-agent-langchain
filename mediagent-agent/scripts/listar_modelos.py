"""
Consulta los modelos disponibles con la API key actual.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"))

import anthropic

client = anthropic.Anthropic()
models = client.models.list()

print("Modelos disponibles con tu API key:")
print("-" * 50)
for m in models.data:
    print(f"  {m.id}")
