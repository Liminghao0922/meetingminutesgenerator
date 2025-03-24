# from transformers import AutoModelForCausalLM, AutoTokenizer
# import os

# # from transformers import TextStreamer


# def optimize_text(
#     pretrained_model_name_or_path: str | os.PathLike[str],
#     user_prompt: str,
#     max_new_tokens=32768,
#     temperature=0.7,
#     top_p=0.9,
# ):
#     # Load model and tokenizer
#     model = AutoModelForCausalLM.from_pretrained(
#         pretrained_model_name_or_path, device_map="auto", torch_dtype="auto"
#     )
#     tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

#     # Prepare input with chat template
#     messages = [
#         {
#             "role": "user",
#             "content": user_prompt,
#         }
#     ]
#     input_ids = tokenizer.apply_chat_template(
#         messages, add_generation_prompt=True, return_tensors="pt"
#     ).to(model.device)

#     # Generate output
#     output_ids = model.generate(
#         input_ids,
#         max_new_tokens=max_new_tokens,
#         temperature=temperature,
#         top_p=top_p,
#         do_sample=True,  # Enable sampling for temperature and top_p
#     )

#     # Decode the output and remove the input prompt
#     # full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#     # input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
#     # optimized_text = full_text[len(input_text):].strip()
#     optimized_text = tokenizer.decode(
#         output_ids.tolist()[0][input_ids.size(1) :], skip_special_tokens=True
#     )

#     return optimized_text
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class TextOptimizer:
    def __init__(self, pretrained_model_name_or_path: str | os.PathLike[str]):
        """Initialize the TextOptimizer with model and system prompt."""
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, device_map="auto", torch_dtype="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        # Load system prompt from file
        prompt_file = os.path.join("prompts", "system_prompt.txt")
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"System prompt file not found at {prompt_file}")
        with open(prompt_file, "r", encoding="utf-8") as f:
            self.system_prompt_template = f.read()

    def optimize_text(
        self,
        raw_text: str,
        domain_terms: str = "",
        misrecognized_words: str = "",
        max_new_tokens=32768,
        temperature=0.7,
        top_p=0.9,
    ):
        """Optimize the provided text using the loaded model and system prompt."""
        # Fill system prompt with domain terms and misrecognized words
        system_prompt = self.system_prompt_template.format(
            domain_terms=domain_terms if domain_terms else "なし",
            misrecognized_words=misrecognized_words if misrecognized_words else "なし"
        )

        # Prepare input with chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        # Generate output
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        # Decode the output and remove the input prompt
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        optimized_text = full_text[len(input_text):].strip()

        return optimized_text