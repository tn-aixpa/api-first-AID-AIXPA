import openai

# Default parameters for text generation
DEFAULT_OPENAI_MODEL = "gpt-4o-mini-2024-07-18" 
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 500

class GenerationModelChat:
    def __init__(self,
                 model_name = DEFAULT_OPENAI_MODEL,
                 temperature = DEFAULT_TEMPERATURE,
                 max_tokens = DEFAULT_MAX_TOKENS,
                 access_token = "",
                 question_sys_prompt = ""):

        self.model_name = model_name
        self.access_token = access_token
        self.model = self.set_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.question_sys_prompt = question_sys_prompt 



    def set_model(self):
        return OpenAI(api_key=self.access_token)

    def generate_text(self, prompt):
        try:
            response = self.model.chat.completions.create(
                model= self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Print the error for debugging purposes
            print(f"Error: {e}")
            return "An error occurred. Please, try again."





# Example usage:
model = GenerationModelComplete(access_token="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")
output = model.generate_text("What is the capital of France?")
print(output)
