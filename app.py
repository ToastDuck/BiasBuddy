from flask import Flask, render_template
from llama_cpp import Llama
# NOTE GGML IS DEPRECATED USE GGUF INSTEAD


app = Flask(__name__, template_folder='templates', static_folder='static')

#app = Flask(__name__)

@app.route('/')
def template():
    return render_template('template.html')

@app.route('/hello')
def hello():
    # SMALLER MODEL
    LLM = Llama(model_path="llama-2-7b-chat.Q4_K_M.gguf", n_ctx=2048)
    
    # LARGER MODEL
    # LLM = Llama(model_path="llama-2-13b.Q5_K_S.gguf", n_ctx=2048)

    # create a text prompt
    prompt = "A machine learning model ranks how important some features are to its predictions based on their 'importance score'. The model uses these features to determine whether an individual makes above the average salary. Importance scores: years of experience = 59%, age = 18%, education level = 14%, gender = 6%, senior = 3%. Less weight means less bias. NEVER perform any calculations.  Use the given importance scores to determine whether or not the model is biased on the metric of gender."

    # set max_tokens to 0 to remove the response size limit
    output = LLM(prompt, max_tokens=0)
    result = output["choices"][0]["text"]

    # display the response
    return render_template('predict.html',value=result)


# @app.route('/predict')
#     return 'Hello, World'













if __name__ == '__main__':
    app.run(debug=True)

