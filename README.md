# Microgpt-Complete-GPT-algorithm
A fork of Andrej Karpathy's Microgpt, a small 200 line python file that shows the "most atomic way to train and inference a GPT in pure, dependency-free Python."
- It is a minimal, educational implementation of a GPT (Generative Pre-trained Transformer) model written in pure Python. It demonstrates the core algorithms behind modern Large Language Models (LLMs) like ChatGPT without using any deep learning libraries like PyTorch or TensorFlow. 
- Because it is written in pure Python without optimization or GPU acceleration, it is significantly slower than standard libraries like PyTorch, but it is excellent for understanding exactly how the math inside a GPT works.
The code implements the following components of a GPT model:
1. **Tokenization**: A simple character-level tokenizer that converts text into integer token IDs and back.
2. **Model Architecture**: A basic implementation of the Transformer architecture, including multi-head self-attention and feedforward layers.
3. **Training Loop**: A simple training loop that uses stochastic gradient descent to optimize the model parameters based on a given dataset.
4. **Inference**: A method to generate text by sampling from the model's output probabilities.

To use the code, save the microgpt.py file and run `python microgpt.py`. It will train the model on a small dataset (in this case, a string of text) and then generate new text based on the learned patterns.

Note that this implementation is not optimized for performance and is intended for educational purposes only. It may take a long time to train and generate text, especially on larger datasets. For practical applications, consider using established libraries like PyTorch or TensorFlow, which are optimized for performance and can leverage GPU acceleration.


