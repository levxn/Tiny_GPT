import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer
from scraper import google_search_and_extract_text
from summarie import summary

import warnings
warnings.filterwarnings('ignore')

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
string = ""

# def summary(items, c):
#     # return summarize(items, word_count=c)
#     summarizer = Summarizer()
#     return summarizer(items, min_length=c)


def new_prompt():
    s = input("Your Entry: ")
    content = google_search_and_extract_text(s)
    worker(content)
    return s


def worker(content):
    for i in range(10):
        j = 0

        question = input("Whats your question ? :")
        while j < len(content):
            try:
                print(answer_question(question, content[j]))
            except:
                try:
                    g = summary(content[j], 500)
                    print(answer_question(question, g))
                except:
                    try:
                        g = summary(content[j], 400)
                        print(answer_question(question, g))
                    except:
                        g = summary(content[j], 300)
                        print(answer_question(question, g))

            w = input("Did This Answer your Question? : ")
            if w == "y":
                print("Thenks")
                d = input("Should we continue ? :")
                if d == "y":
                    break
                else:
                    new_prompt()
            else:
                j += 1

            if j+3 == len(content):
                print("\n\n\nUnable to find details... Sorry")
                new_prompt()

def answer_question(question, answer_text):
    input_ids = tokenizer.encode(question, answer_text)
    sep_index = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    assert len(segment_ids) == len(input_ids)

    outputs = model(torch.tensor([input_ids]),
                    token_type_ids=torch.tensor([segment_ids]),
                    return_dict=True)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    answer = tokens[answer_start]

    for i in range(answer_start + 1, answer_end + 1):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]

    return answer


new_prompt()
