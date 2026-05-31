from llama_cpp import Llama

llm = Llama(model_path="models/AceGPT-7B-chat.Q4_K_M.gguf",
            n_ctx=512, n_threads=4, verbose=False)
print("LOADED OK")
out = llm("Hello", max_tokens=3, echo=False)
print("INFERENCE OK:",
      out.get("choices", [{}])[0].get("text", "")[:50])
