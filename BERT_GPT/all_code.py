import re
from string import punctuation
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import textwrap
from transformers import BertTokenizer
from transformers import BertForQuestionAnswering
import torch
import streamlit as st
from streamlit_chat import message
import openai
import time

st.title("🤖 QA-Model A-K-L")
# question = "How many parameters does BERT-large have?"
# answer_text = "BERT-large is really big... it has 24-layers and an embedding size of 1,024, for a total of 340M parameters! Altogether it is 1.34GB, so expect it to take a couple minutes to download to your Colab instance."

# ---------------------------------------------------------------------------

model = BertForQuestionAnswering.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad')


@st.cache_data
def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example through the model.
    outputs = model(torch.tensor([input_ids]),  # The tokens representing our input text.
                    # The segment IDs to differentiate question from answer_text
                    token_type_ids=torch.tensor([segment_ids]),
                    return_dict=True)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    s = '"' + answer + '"'
    return tokenizer, model, s


# -----------------------------------------------------------------------------
# Wrap text to 80 characters.
wrapper = textwrap.TextWrapper(width=80)
bert_abstract = "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018a; Radford et al., 2018), BERT is designed to pretrain deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement)."

# s = ''
# question = st.text_area('Enter Question: ')
# button = st.button("Get Answer")

# tokenizer, model, s = answer_question(question, bert_abstract)
# -----------------------------------------------------------------------------


def generate_response(prompt):
    if prompt:
        test_sample = tokenizer(
            [question], padding=True, truncation=True, max_length=512, return_tensors='pt')
        output = model(**test_sample)
    return s


# ----------------------------------------------------------------------------
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"

if 'past' not in st.session_state:
    st.session_state.past = []

if "messages" not in st.session_state:
    st.session_state.messages = []

question = st.chat_input("What is up?")
tokenizer, model, s = answer_question(question, bert_abstract)


if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # with st.chat_message("assistant"):
    #     message_placeholder = st.empty()
    #     full_response = ""
    #     assistant_response = s
    #     st.markdown(assistant_response)
    #     message_placeholder.markdown(full_response)
    # st.session_state.messages.append(
    #     {"role": "assistant", "content": assistant_response})

    with st.chat_message("assistant"):
        assistant_response = s
        # st.markdown(assistant_response)

        message_placeholder = st.empty()
        # Adjust typing speed as needed (in seconds per character)
        typing_speed = 0.03

        full_response = ""
        for char in assistant_response:
            full_response += char
            message_placeholder.markdown(full_response + "▌")
            time.sleep(typing_speed)

        message_placeholder.text(full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_response})

for s in st.session_state.messages:
    with st.chat_message(s["role"]):
        st.markdown(s["content"])

# if question:
#     output = generate_response(question)
#     st.session_state.past.append(question)
    # st.session_state.generated.append(output)
###
###


# # -----------------------------------------------------------------------------
# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("Enter Question"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()
#         full_response = ""
#         for response in openai.ChatCompletion.create(
#             model=st.session_state["openai_model"],
#             messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ],
#             stream=True,
#         ):
#             full_response += response.choices[0].delta.get("content", "")
#             message_placeholder.markdown(full_response + "▌")
#         message_placeholder.markdown(full_response)
#     st.session_state.messages.append(
#         {"role": "assistant", "content": full_response})


# from scraper import google_search_and_extract_text


# def summary(text):
#     extra_words = list(STOP_WORDS) + list(punctuation) + ['\n']
#     nlp = spacy.load('en_core_web_sm')
#     # doc = ' '.join(google_search_and_extract_text(text))
#     docx = nlp(text)
#     all_words = [word.text for word in docx]
#     Freq_word = {}

#     for w in all_words:
#         w1 = w.lower()
#         if w1 not in extra_words and w1.isalpha():
#             if w1 in Freq_word.keys():
#                 Freq_word[w1] += 1
#             else:
#                 Freq_word[w1] = 1

#     val = sorted(Freq_word.values())
#     max_freq = val[-3:]

#     for word in Freq_word.keys():
#         Freq_word[word] = (Freq_word[word] / max_freq[-1])

#     sent_strength = {}
#     for sent in docx.sents:
#         for word in sent:
#             if word.text.lower() in Freq_word.keys():
#                 if sent in sent_strength.keys():
#                     sent_strength[sent] += Freq_word[word.text.lower()]
#                 else:
#                     sent_strength[sent] = Freq_word[word.text.lower()]
#             else:
#                 continue

#     top_sentences = (sorted(sent_strength.values())[::-1])
#     top30percent_sentence = int(0.3 * len(top_sentences))
#     top_sent = top_sentences[:top30percent_sentence]

#     summary = []
#     for sent, strength in sent_strength.items():
#         if strength in top_sent:
#             summary.append(sent)
#         else:
#             continue

#     s = []
#     for sentence in summary:
#         s.append(re.sub(r'\s+', ' ', sentence.text))

#     return ' '.join(s)

# Example usage:
# ques = input('type: ')
# summary = generate_summary(ques)
# for sentence in summary:
#     print(re.sub(r'\s+', ' ', sentence.text))
