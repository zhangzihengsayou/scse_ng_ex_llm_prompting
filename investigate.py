## Import the necessary modules
import os
import json
import ollama
from parse_data import load_items, get_unclaimed_items, save_result


## Build your prompt based on the description the user provides 
## and the items that are available in the lost-and-found database.
## The model must follow the rules listed in the README file
## The function should return the system prompt and the user prompt.
def build_prompt(description, available_items):
    system_prompt = """You are a campus lost-and-found assistant. Your task is to match a user's description of a lost item against a database of found items.

Rules:
- Use ONLY the given JSON file of found items.
- Not all details of an item must match to be a possible match.
- Return ONLY JSON, with exactly this structure:
{
    "matches": ["ITEM_ID"],
    "confidence": "LOW"
}
- "matches" contains all possible matching item IDs.
- "confidence" must be exactly one of: LOW, MEDIUM, HIGH.
- If there is no match, return an empty list for "matches".
- Do not include any explanation, markdown, or extra text. Only the JSON object."""

    user_prompt = f"""The user lost: "{description}"

Here are the available found items (JSON):
{json.dumps(available_items, indent=2, ensure_ascii=False)}

Return the JSON result with possible matches and confidence."""

    return system_prompt, user_prompt


## Logic to ask Qwen for all the possible matches based on the system prompt and user prompt.
## The function should return the response from Qwen.
## The model name can be overridden via the QWEN_MODEL environment variable.
def ask_qwen(system_prompt, user_prompt):
    model = os.environ.get("QWEN_MODEL", "qwen2.5:1.5b")
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response["message"]["content"]


## Logic to parse the response from Qwen and return the result.
## You may need to use json.loads() to convert the response string into a suitable Python data structure.
def parse_response(response_text):
    text = response_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


## Logic to validate the result returned by Qwen.
## It should check if the result is a dictionary, contains the keys "matches" and "confidence",
## and that the values are of the correct type.
## If everything is correct, then it should check if the item IDs in the "matches" list are valid IDs.
def validate_result(result, available_items):
    if not isinstance(result, dict):
        return False
    if "matches" not in result or "confidence" not in result:
        return False
    if not isinstance(result["matches"], list):
        return False
    if result["confidence"] not in ("LOW", "MEDIUM", "HIGH"):
        return False
    if not all(isinstance(m, str) for m in result["matches"]):
        return False
    valid_ids = {item["id"] for item in available_items}
    if not all(m in valid_ids for m in result["matches"]):
        return False
    return True


## Logic to display the matches found by Qwen in a user-friendly format.
def display_matches(result, available_items):
    print("\nMATCH RESULT")
    print("-" * 50)
    print(f"Confidence: {result['confidence']}")
    print()

    if not result["matches"]:
        print("No matches found.")
        print("Possible matches: []")
        return

    print("Possible matches:")
    items_by_id = {item["id"]: item for item in available_items}
    for match_id in result["matches"]:
        item = items_by_id.get(match_id)
        if item:
            print()
            print(f"ID: {item['id']}")
            print(f"Item: {item['item']}")
            print(f"Color: {item['color']}")
            print(f"Location: {item['location']}")
            print(f"Date found: {item['date']}")


## Control center for the entire program.
def main():
    print("CAMPUS LOST-AND-FOUND ASSISTANT")
    print("=" * 50)
    description = input("Describe the item you lost: ").strip()

    print("\nSearching for possible matches...\n")

    items = load_items("found_items.json")
    unclaimed = get_unclaimed_items(items)

    system_prompt, user_prompt = build_prompt(description, unclaimed)
    response_text = ask_qwen(system_prompt, user_prompt)
    result = parse_response(response_text)

    if result is None or not validate_result(result, unclaimed):
        # Fallback to an empty result if the model response is invalid
        result = {"matches": [], "confidence": "LOW"}

    display_matches(result, unclaimed)

    save_result(result, "output/match_result.json")
    print("\nResult saved to output/match_result.json")


if __name__ == "__main__":
    main()
