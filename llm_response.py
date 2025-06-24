# llm_response.py

from transformers import pipeline
import re

# === Load FLAN-T5 model ===
try:
    # Consider using flan-t5-base or flan-t5-small for lower RAM use on free tier
    # generator = pipeline("text2text-generation", model="google/flan-t5-base")  # safe for Streamlit Cloud
    generator = pipeline("text2text-generation", model="google/flan-t5-large")  # may crash on free plan but try
except Exception as e:
    generator = None
    print(f"⚠️ Failed to load LLM model: {e}")

# === Input & Output Cleaning ===
def sanitize_input(text):
    text = str(text).strip()
    return re.sub(r"[^a-zA-Z0-9,.\-? %+/=$mgMG()]", "", text)

def postprocess_output(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if text and not text.endswith("."):
        text += "."
    return text[0].upper() + text[1:] if text else ""

# === Main LLM Prompt ===
def generate_initial_response(condition, confidence, age, skin_type, symptoms):
    if generator is None:
        return "⚠️ AI model not available."

    condition = sanitize_input(condition)
    skin_type = sanitize_input(skin_type)
    symptoms = sanitize_input(symptoms)
    try:
        age = int(age)
    except Exception:
        age = "unknown"

    prompt = (
        "You are a helpful medical assistant.\n\n"
        f"A patient has been diagnosed with {condition} (confidence {confidence*100:.1f}%).\n"
        f"Patient age: {age}, Skin type: {skin_type}.\n"
        f"Reported symptoms: {symptoms}.\n\n"
        "Provide a concise explanation of the condition and recommended next steps or treatment advice."
    )

    try:
        output = generator(prompt, max_length=150, num_beams=5, early_stopping=True)
        generated_text = output[0]['generated_text']
        answer = generated_text[len(prompt):].strip()
        if not answer:
            answer = generated_text.strip()
        return postprocess_output(answer)
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

# === Follow-Up Prompt Template ===
ONE_SHOT_TEMPLATE = """
You are a helpful medical assistant that provides clear, medically-informed treatment options and advice for skin conditions.

Example:

Condition: Atopic Dermatitis (confidence: 91%)
Age: 28
Skin Type: Dry
Question: What is the best treatment option for this condition?
Answer: Treatment for Atopic Dermatitis includes regular use of emollients and moisturizers to keep the skin hydrated, and topical corticosteroids to reduce inflammation during flare-ups. Lifestyle advice includes avoiding known triggers like allergens, stress, and harsh soaps. If symptoms persist or cause discomfort, a dermatologist should be consulted.

Now answer this case:

Condition: {condition} (confidence: {confidence}%)
Age: {age}
Skin Type: {skin_type}
Question: {followup_question}
Answer:
"""

def generate_followup_response(condition, confidence, age, skin_type, followup_question):
    if generator is None:
        return "⚠️ AI model not available."

    condition = sanitize_input(condition)
    skin_type = sanitize_input(skin_type)
    followup_question = sanitize_input(followup_question)

    try:
        age = int(age)
    except Exception:
        age = "unknown"

    prompt = ONE_SHOT_TEMPLATE.format(
        condition=condition,
        confidence=round(confidence * 100),
        age=age,
        skin_type=skin_type,
        followup_question=followup_question
    )

    try:
        output = generator(prompt, max_length=200, num_beams=5, no_repeat_ngram_size=3, early_stopping=True)
        return postprocess_output(output[0]['generated_text'])
    except Exception as e:
        return f"⚠️ Error generating follow-up response: {str(e)}"
