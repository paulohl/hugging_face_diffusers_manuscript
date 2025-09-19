###### This example demonstrates how to utilize GPT-3 to develop a conversational agent.######
###### These agents will assist with creative writing tasks.                            ######
############################################################################################## 
from transformers import GPT3Tokenizer, GPT3Model, pipeline 

# Load pre-trained GPT-3 model 
tokenizer = GPT3Tokenizer.from_pretrained('gpt3') 
model = GPT3Model.from_pretrained('gpt3') 

# Setup text generation pipeline 
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer) 

# Generate dialogue response 
dialogue_prompt = "Customer: I am unable to access my account. Help!" 
dialogue_response = text_generator(dialogue_prompt, max_length=50, num_return_sequences=1) 
print("Dialogue Response:", dialogue_response[0]['generated_text']) 

# Generate creative writing 
creative_prompt = "Write a poem about the ocean." 
poem = text_generator(creative_prompt, max_length=100, num_return_sequences=1) 
print("Generated Poem:", poem[0]['generated_text']) 
