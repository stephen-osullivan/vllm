"""
simple self contained script that instantiates a model and runs the below prompts through it.

"""
from dotenv import load_dotenv
from vllm import LLM, SamplingParams

load_dotenv()

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "How can I write python code to swap 2 numbers?"
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=100)

# Create an LLM.
#llm = LLM(model="facebook/opt-125m")
#llm = LLM(model="google/gemma-2b-it")
llm = LLM(model="TheBloke/Llama-2-7b-Chat-AWQ", quantization="AWQ")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")