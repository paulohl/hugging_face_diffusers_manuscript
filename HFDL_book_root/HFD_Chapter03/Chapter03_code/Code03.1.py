###### This code demonstrates how to use the Hugging Face Diffusion library for text generation tasks.    ######
###### We use a pre-trained GPT model for its performance in generating coherent and context-rich text.   ######
###### This example is a guidance to set up the model, preparing the input data,                          ######
###### and running the generation process to create grammatically correct and contextually relevant text. ######
################################################################################################################

from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline 

# Load pre-trained GPT-2 model and tokenizer 
tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 
model = GPT2LMHeadModel.from_pretrained('gpt2') 

# Initialize text generation pipeline 
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer) 

# Generate text based on a prompt 
prompt = "In a distant future, humanity has ventured far into the cosmos" 
generated_texts = text_generator(prompt, max_length=100, num_return_sequences=1) 

for generated_text in generated_texts: 
    print(generated_text['generated_text'])
  
