from llama_cpp import Llama
# NOTE GGML IS DEPRECATED USE GGUF INSTEAD

# SMALLER MODEL
LLM = Llama(model_path="llama-2-7b-chat.Q4_K_M.gguf", n_ctx=2048)
  
# LARGER MODEL
# LLM = Llama(model_path="llama-2-13b.Q5_K_S.gguf", n_ctx=2048)

# create a text prompt
prompt = "A machine learning model ranks how important some features are to its predictions based on their 'importance score'. The model uses these features to determine whether an individual makes above the average salary. Importance scores: years of experience = 59%, age = 18%, education level = 14%, gender = 6%, senior = 3%.  Use the given importance scores to determine whether or not the model is biased on the metric of gender. Lower importance score normally means less discrimination. NEVER perform any calculations."

# set max_tokens to 0 to remove the response size limit
output = LLM(prompt, max_tokens=0)

# display the response
print(output["choices"][0]["text"])
