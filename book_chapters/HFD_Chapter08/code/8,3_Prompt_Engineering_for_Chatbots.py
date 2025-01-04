# This example shows the use of prompt engineering to tailor a model’s responses to user queries by structuring input prompts.

def generate_prompt(user_input):
    return f"The user says: '{user_input}'. Respond with a helpful reply."

response = model.generate(generate_prompt("I need help resetting my password."))
