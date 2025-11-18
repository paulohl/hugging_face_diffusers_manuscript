# Example: Prompt engineering for chatbots:

def generate_prompt(user_input):
    return f"The user says: '{user_input}'. Respond with a helpful reply."

response = model.generate(generate_prompt("I need help resetting my password."))
