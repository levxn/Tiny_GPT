# Tiny BERT-GPT

A minimalist implementation of a Transformer-based language model inspired by BERT and GPT architectures. This project is designed for experimentation, education, and understanding the core building blocks of Transformer models using PyTorch and Flask.

## 📁 Project Structure

BERT_GPT/
├── ak_bert.py # Core transformer model implementation
├── le_ak_bert.py # Variant of the core model with slight modifications
├── all_code.py # Combined training script with vocabulary and model
├── app.py # Flask app for chatbot interaction
├── chatbot.ipynb # Jupyter notebook for interactive experimentation
├── input.txt # Sample training data
├── nlp.py # Tokenization and NLP utilities
├── scraper.py # Web scraping tool (e.g., Wikipedia)
├── summarie.py # Text summarization using the model
├── v2.py / v3.py # Experimental variants of the training/model pipeline


## 🔧 Installation

Make sure Python 3.7+ is installed along with PyTorch and Flask.

```bash
git clone https://github.com/levxn/Tiny_GPT.git
cd Tiny_GPT/BERT_GPT
pip install torch flask

🚀 Getting Started
1. Training the Transformer
Train a simple transformer-based language model using:
python all_code.py

This script:

Tokenizes input.txt

Builds a vocabulary

Trains the model using a simple training loop

Saves model weights

2. Running the Web Chatbot
To launch a basic chatbot using Flask:
python app.py

Then open http://localhost:5000 in your browser to interact with the model.

3. Notebook Exploration
For an interactive demo using a notebook:

Open chatbot.ipynb with Jupyter or VS Code. It includes:

Data processing

Model instantiation

Training logic

Sample inference

🧠 Core Components
ak_bert.py
Implements a lightweight BERT-style model from scratch:

Multi-head self-attention

Transformer block

Embeddings

Final prediction layer

nlp.py
Includes utility functions for:

Word tokenization

Padding sequences

Vocabulary indexing

scraper.py
Scrapes text content (e.g., Wikipedia) to expand training data or create summaries.

summarie.py
Applies the trained model to perform rudimentary text summarization.

🧪 Experimental Files
le_ak_bert.py, v2.py, v3.py
Alternate versions of the model or training logic.
Use these to explore variants or extend functionality.

📚 Example Training Text
The file input.txt includes raw training text. You can replace this with your own dataset to retrain or fine-tune the model.

❗ Limitations
Not optimized for production or large datasets

No GPU acceleration enabled by default

Trained on extremely limited data

This is an educational project, not a full-scale NLP solution.

📜 License
This project is under the MIT License. You are free to use and modify it for learning or prototyping.

Feel free to copy and paste the content directly into your `README.md` file. Let me know if you need further modifications!
