# from spacy.lang.en.stop_words import STOP_WORDS
# import spacy
# # from scraper import google_search_and_extract_text
# from string import punctuation
# import re

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

# # Example usage:
# # ques = input('type: ')
# # summary = generate_summary(ques)
# # for sentence in summary:
# #     print(re.sub(r'\s+', ' ', sentence.text))

from transformers import BertTokenizer, BertLMHeadModel
import torch


def summary(text, max_tokens):
    # Load a BERT model and tokenizer for text summarization
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertLMHeadModel.from_pretrained(model_name)

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_tokens, padding="max_length")

    # Generate a summary
    summary_ids = model.generate(inputs["input_ids"])

    # Decode the summary
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary_text


# # Example usage:
# input_text = "Your long input text here."
# max_tokens = 512
# processed_text = summary(input_text, max_tokens)
# print(processed_text)
