###### This script demonstrates how to use the GPT-2 model, a variant of the GPT, for text generation.   ######               
###### GPT-2 strrenth is producing coherent and contextually relevant text passages.                     ######
###### We use a pre-trained version to demonstrate how to generate text based on a given prompt.         ######
###############################################################################################################

from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline 

# Load the model and tokenizer for GPT-2 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium') 
model = GPT2LMHeadModel.from_pretrained('gpt2-medium') 

# Setup the pipeline for text generation 
generator = pipeline('text-generation', model=model, tokenizer=tokenizer) 

# Generate text from a prompt 
prompt = "The future of AI in medicine is" 
generated_text = generator(prompt, max_length=50, num_return_sequences=1) 

print("Generated Text:") 
for i, text in enumerate(generated_text): 
    print(f"{i+1}: {text['generated_text']}") 
