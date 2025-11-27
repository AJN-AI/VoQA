# The script for analyse GQA, TextVQA and SQA. 
import argparse
import json
import csv
import os
from eval_textvqa import eval_single
from process_answer import str2bool
from process_answer import split_sentences

########## Please set the corresponding parameters here ###########
QAA_CSV_PATH = '/path/to/save/QAA.csv'
OUTPUT_TYPE_CSV_PATH = '/path/to/save/Output_type.csv'

# define SFT type for models
sft_type_dict = {
    'InternVL3-1B-QA': 'R-QA',
    'InternVL3-1B-RQA': 'RQA',
    'InternVL3-1B-QA_only': 'QA',
    'InternVL3-1B-R-QRA': 'R-QRA',
    'InternVL3-1B-RQRA': 'RQRA', 
    'InternVL3-1B-QRA': 'QRA',
}

# define split word for model
split_word_dict = {
    # internvl3-1b-pretrained
    'InternVL3-1B-QRA': '\nassistant\n',
    'InternVL3-1B-RQA': '\nassistant\n',
    'InternVL3-1B-R-QRA': '\nassistant\n',
    'InternVL3-1B-RQRA': '\nassistant\n', 

    # tinyllava
    'TinyLLaVA-Qwen2-0.5B-SigLIP-QRA': 'ASSISTANT:',
    'TinyLLaVA-Qwen2-0.5B-SigLIP-QRA-HELPER': 'HELPER:',
    'TinyLLaVA-Qwen2-0.5B-SigLIP-QRA-CAT': 'CAT',
}

########## Please set the corresponding parameters here ###########

def compute_edit_distance(ref: str, pred: str) -> float:
    m, n = len(ref), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == pred[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    
                    dp[i][j-1] + 1,    
                    dp[i-1][j-1] + 1   
                )
    
    edit_distance = dp[m][n]
    if len(ref) == 0:
        return 0.0
    return 1 - edit_distance / len(ref)



def process_sqa(sqa_pred_answers, sqa_inference_answers, save_file, model_name, split_word='ASSISTANT'):
    sft_type = sft_type_dict[model_name]

    question_id_dict = {
        'correct': [],
        'incorrect': []
    }
    
    with open(sqa_pred_answers, 'r') as f:
        data = json.load(f)
        
    for result_type in ['correct', 'incorrect']:
        for item in data[result_type]:
            question_id_dict[result_type].append(item['question_id'])
    
    similarity_scores = {
        'correct': {},
        'incorrect': {}
    }

    analysis_data = []
    correct_st_cnt = 0 
    correct_other_cnt = 0
    incorrect_st_cnt = 0
    incorrect_other_cnt = 0
    st_cnt = 0
    other_cnt = 0
    correct_cnt = 0

    with open(sqa_inference_answers, 'r') as f:
        for line in f:
            data = json.loads(line)
            question_id = data['question_id']
            
            real_question = data['prompt'].replace('<image>', '').replace('\n', ' ').strip()
            
            if data['text'].startswith('USER: \n\n ASSISTANT: QUESTION:'):
                split_parts_text = data['text'][29:]
            elif data['text'].startswith('USER: \n\n ASSISTANT:'):
                split_parts_text = data['text'][19:]
            elif  data['text'].startswith('USER: \n\n'):
                split_parts_text = data['text'][8:]

            elif data['text'].startswith('QUESTION:'):
                split_parts_text = data['text'][9:]
            else:
                split_parts_text = data['text']
            split_parts_text = split_parts_text.replace('<image>', '')
            # split_parts_text = split_parts_text.replace('\n', ' ')
            split_parts_text = split_parts_text.strip()

            begin_with_split_word = False
            if sft_type in ['RQA', 'RQRA'] and split_parts_text.startswith('assistant\n'):
                begin_with_split_word = True
                split_parts_text = split_parts_text[10:]

            if sft_type not in ['R-QA', 'QA', 'RQA']:
                model_split_word = split_word_dict[model_name]
            else:
                model_split_word = split_word
            split_parts = split_parts_text.split(f'{model_split_word}')

            if sft_type in ['R-QA', 'QA', 'RQA']:
                if sft_type in ['R-QA', 'QA'] and len(split_parts) == 1:
                    pred_question = split_parts[0].replace('\n', ' ').strip()
                elif sft_type in ['RQA'] and begin_with_split_word and len(split_parts) == 1:
                    pred_question = split_parts[0].replace('\n', ' ').strip()
                else:
                    pred_question = split_parts_text.replace('\n', ' ').strip()

                split_sentences_lst = split_sentences(pred_question)
                
                input_sentences_lst = []
                current = []
                similarity_score_lst = []
                max_similarity_score = 0
                max_sentence_num = 0
                for j, s in enumerate(split_sentences_lst):
                    current.append(s)
                    current_sentence = " ".join(current)
                    input_sentences_lst.append(current_sentence)
                    similarity_score = compute_edit_distance(real_question, current_sentence)
                    # modified
                    if similarity_score < 0 :
                        # print(similarity_score, question_list[i])
                        similarity_score = 0
                    if max_similarity_score < similarity_score:
                        max_similarity_score = similarity_score
                        max_sentence_num = j + 1

                    similarity_score_lst.append(similarity_score)
                
                similarity = max_similarity_score

                sentence_num = len(split_sentences_lst)
                if len(split_parts) == 1 and sentence_num > max_sentence_num and max_similarity_score > 0: # Q+A
                    if (sft_type in ['RQA'] and begin_with_split_word) or sft_type in ['R-QA',  'QA', ]:
                        output_type = sft_type
                        st_cnt += 1
                    else:
                        output_type = 'others'
                        other_cnt += 1                        
                else:
                    output_type = 'others'
                    other_cnt += 1

            elif sft_type in ['R-QRA', 'QRA', 'RQRA']:
                similarity_score_lst = None
                max_sentence_num = None
        
                if sft_type in ['R-QRA', 'QRA'] and len(split_parts) == 2:
                    pred_question = split_parts[0].replace('\n', ' ').strip()
                    st_cnt += 1 
                    output_type = sft_type
                
                elif sft_type in ['RQRA'] and begin_with_split_word and len(split_parts) == 2:
                    pred_question = split_parts[0].replace('\n', ' ').strip()
                    st_cnt += 1 
                    output_type = sft_type                    
                else: 
                    pred_question = split_parts_text.replace('\n', ' ').strip()
                    other_cnt += 1
                    output_type = 'others'
                

                # print([real_question, pred_question])
                similarity = compute_edit_distance(real_question, pred_question)

                # modified
                if similarity < 0 :
                    # print(similarity, data['text'])
                    similarity = 0
            else:
                print("Wrong SFT Type!")
            
            if question_id in question_id_dict['correct']:
                similarity_scores['correct'][question_id] = similarity
                is_true = True
                if output_type != 'others':
                    correct_st_cnt += 1
                else:
                    correct_other_cnt += 1
                correct_cnt += 1
            elif question_id in question_id_dict['incorrect']:
                similarity_scores['incorrect'][question_id] = similarity
                is_true = False
                if output_type != 'others':
                    # print('incorrect_st_cnt')
                    incorrect_st_cnt += 1
                else:
                    # print('incorrect_other_cnt')
                    incorrect_other_cnt += 1
            
            jsonl_dict ={
                "part_num:": len(split_parts),
                "similarity_score": similarity,
                "output_type": output_type,
                "is_true": is_true,
                "question_id": question_id,
                "original_output": data['text'],
                "split_part_lst": split_parts,
                "original_question": real_question,
                # QA-SFT 
                "similarity_score_lst": similarity_score_lst,
                "max_score_sentence_num": max_sentence_num,
            }
            # print(jsonl_dict)
            analysis_data.append(jsonl_dict)

    with open(args.analysis_file, "w", encoding="utf-8") as f:
        for entry in analysis_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    correct_avg = sum(similarity_scores['correct'].values()) / len(similarity_scores['correct']) if similarity_scores['correct'] else 0
    incorrect_avg = sum(similarity_scores['incorrect'].values()) / len(similarity_scores['incorrect']) if similarity_scores['incorrect'] else 0

    print(f"correct answer average QAA: {correct_avg * 100:.2f}")
    print(f"incorrect answer average QAA: {incorrect_avg * 100:.2f}")
    
    if save_file:
        with open(save_file, 'w') as f:
            json.dump(similarity_scores, f, indent=2)

    final_acc = correct_cnt / (st_cnt + other_cnt)
    print(f"SQA ACC:{final_acc * 100:.2f}")
    print(f"Total: There are a total of {st_cnt} pieces of data in the {sft_type} format. The proportion is {(st_cnt / (st_cnt + other_cnt)) * 100:.2f}.")
    print(f"Total: There are a total of {other_cnt} pieces of data in the 'others' format. The proportion is {(other_cnt / (st_cnt + other_cnt)) * 100:.2f}.")

    correct_QAA_percent = correct_st_cnt / (correct_st_cnt + correct_other_cnt) if correct_st_cnt + correct_other_cnt else -1
    correct_others_percent = correct_other_cnt / (correct_st_cnt + correct_other_cnt) if correct_st_cnt + correct_other_cnt else -1 
    incorrect_QAA_percent = incorrect_st_cnt / (incorrect_st_cnt + incorrect_other_cnt) if incorrect_st_cnt + incorrect_other_cnt else -1
    incorrect_others_percent = incorrect_other_cnt / (incorrect_st_cnt + incorrect_other_cnt) if incorrect_st_cnt + incorrect_other_cnt else -1

    print(f"Correct answer: There are a total of {correct_st_cnt} pieces of data in the {sft_type} format. The proportion is {correct_QAA_percent * 100:.2f}.")
    print(f"Correct answer: There are a total of {correct_other_cnt} pieces of data in the 'others' format. The proportion is {correct_others_percent * 100:.2f}.")
    print(f"Incorrect answer: There are a total of {incorrect_st_cnt} pieces of data in the {sft_type} format. The proportion is {incorrect_QAA_percent * 100:.2f}.")
    print(f"Incorrect answer: There are a total of {incorrect_other_cnt} pieces of data in the 'others' format. The proportion is {incorrect_others_percent * 100:.2f}.")

    print(f"Total: All samples average QAA is {(correct_avg * final_acc + incorrect_avg * (1 - final_acc)) * 100:.2f}")
    print(f"Total: In all samples, The proportion of {sft_type} type is {(correct_QAA_percent * final_acc + incorrect_QAA_percent * (1 - final_acc)) * 100:.2f}")
    print(f"Total: In all samples, The proportion of 'others' type is {(correct_others_percent * final_acc + incorrect_others_percent * (1 - final_acc)) * 100:.2f}")

    # QRA: model_name sft_type task correct incorrect total
    QAA_header = ['model_name', 'sft_type', 'task', 'correct_QAA', 'incorrect_QAA', 'total_QAA', 'acc']
    QAA_dict = {
        'model_name': model_name,
        'sft_type': sft_type,
        'task': 'sqa',
        'correct_QAA': correct_avg * 100,
        'incorrect_QAA': incorrect_avg * 100,
        'total_QAA': (correct_avg * final_acc + incorrect_avg * (1 - final_acc)) * 100,
        'acc': final_acc * 100
    }
    QAA_file_exists = os.path.exists(QAA_CSV_PATH)
    with open(QAA_CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=QAA_header)

        if not QAA_file_exists:
            writer.writeheader()

        writer.writerow(QAA_dict)

    # Output_type: model_name sft_type task correct(T/F) incorrect(T/F) total(T/F) 
    output_type_header = ['model_name', 'sft_type', 'task', 'correct_True_output_type', 'correct_False_output_type',
     'incorrect_True_output_type', 'incorrect_False_output_type', 'total_True_output_type', 'total_False_output_type']
    output_type_dict = {
        'model_name': model_name,
        'sft_type': sft_type,
        'task': 'sqa',
        'correct_True_output_type': correct_QAA_percent * 100,
        'correct_False_output_type': correct_others_percent * 100,
        'incorrect_True_output_type': incorrect_QAA_percent * 100,
        'incorrect_False_output_type': incorrect_others_percent * 100,
        'total_True_output_type': (correct_QAA_percent * final_acc + incorrect_QAA_percent * (1 - final_acc)) * 100,
        'total_False_output_type': (correct_others_percent * final_acc + incorrect_others_percent * (1 - final_acc)) * 100
    }
    
    output_type_file_exists = os.path.exists(OUTPUT_TYPE_CSV_PATH)

    with open(OUTPUT_TYPE_CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=output_type_header)

        if not output_type_file_exists:
            writer.writeheader()
        
        writer.writerow(output_type_dict)


def process_textvqa(textvqa_question_id_scores_list, textvqa_inference_answers, save_file, final_acc, model_name, split_word='ASSISTANT'):
    sft_type = sft_type_dict[model_name]

    with open(textvqa_question_id_scores_list, 'r') as f:
        score_to_qids = json.load(f)
    
    qid_to_score = {}
    
    for score_str in score_to_qids:
        score = round(float(score_str), 1)
        qids = score_to_qids[score_str]
        for qid in qids:
            qid_to_score[qid] = score
    
    similarity_scores = {str(score): {} for score in set(qid_to_score.values())}
    output_type_scores = {str(score): {} for score in set(qid_to_score.values())}
    
    analysis_data = []
    st_cnt = 0 
    other_cnt = 0
    score_sum = 0
    # st_score_sum = 0

    with open(textvqa_inference_answers, 'r') as f:
        for line in f:
            data = json.loads(line)
            question_id = data['question_id']
            
            if question_id in qid_to_score:
                real_question = data['prompt'].replace('<image>', '').replace('\n', ' ').strip()
                
                if data['text'].startswith('USER: \n\n ASSISTANT: QUESTION:'):
                    split_parts_text = data['text'][29:]
                elif data['text'].startswith('USER: \n\n ASSISTANT:'):
                    split_parts_text = data['text'][19:]
                elif  data['text'].startswith('USER: \n\n'):
                    split_parts_text = data['text'][8:]

                elif data['text'].startswith('QUESTION:'):
                    split_parts_text = data['text'][9:]
                else:
                    split_parts_text = data['text']
                split_parts_text = split_parts_text.replace('<image>', '')
                # split_parts_text = split_parts_text.replace('\n', ' ')
                split_parts_text = split_parts_text.strip()

                begin_with_split_word = False
                if sft_type in ['RQA', 'RQRA'] and split_parts_text.startswith('assistant\n'):
                    begin_with_split_word = True
                    split_parts_text = split_parts_text[10:]
                
                if sft_type not in ['R-QA', 'QA', 'RQA']:
                    model_split_word = split_word_dict[model_name]
                else:
                    model_split_word = split_word
                split_parts = split_parts_text.split(f'{model_split_word}')

                if sft_type in ['R-QA', 'QA', 'RQA']:
                    if sft_type in ['R-QA', 'QA'] and len(split_parts) == 1:
                        pred_question = split_parts[0].replace('\n', ' ').strip()
        
                    elif sft_type in ['RQA'] and begin_with_split_word and len(split_parts) == 1:
                        pred_question = split_parts[0].replace('\n', ' ').strip()
                    else:
                        pred_question = split_parts_text.replace('\n', ' ').strip()

                    split_sentences_lst = split_sentences(pred_question)
                    
                    input_sentences_lst = []
                    current = []
                    similarity_score_lst = []
                    max_similarity_score = 0
                    max_sentence_num = 0
                    for j, s in enumerate(split_sentences_lst):
                        current.append(s)
                        current_sentence = " ".join(current)
                        input_sentences_lst.append(current_sentence)
                        similarity_score = compute_edit_distance(real_question, current_sentence)
                        # modified
                        if similarity_score < 0 :
                            # print(similarity_score, question_list[i])
                            similarity_score = 0
                        if max_similarity_score < similarity_score:
                            max_similarity_score = similarity_score
                            max_sentence_num = j + 1

                        similarity_score_lst.append(similarity_score)
                    
                    similarity = max_similarity_score

                    sentence_num = len(split_sentences_lst)
                    if len(split_parts) == 1 and sentence_num > max_sentence_num and max_similarity_score > 0: # Q+A
                        if (sft_type in ['RQA'] and begin_with_split_word) or sft_type in ['R-QA',  'QA', ]:
                            output_type = sft_type
                            st_cnt += 1
                        else:
                            output_type = 'others'
                            other_cnt += 1                        
                    else:
                        output_type = 'others'
                        other_cnt += 1

                elif sft_type in ['R-QRA', 'QRA', 'RQRA']:
                    similarity_score_lst = None
                    max_sentence_num = None
            
                    if sft_type in ['R-QRA', 'QRA'] and len(split_parts) == 2:
                        pred_question = split_parts[0].replace('\n', ' ').strip()
                        st_cnt += 1 
                        output_type = sft_type
                
                    elif sft_type in ['RQRA'] and begin_with_split_word and len(split_parts) == 2:
                        pred_question = split_parts[0].replace('\n', ' ').strip()
                        st_cnt += 1 
                        output_type = sft_type                    
                    else: 
                        pred_question = split_parts_text.replace('\n', ' ').strip()
                        other_cnt += 1
                        output_type = 'others'
                    
                    similarity = compute_edit_distance(real_question, pred_question)

                    # modified
                    if similarity < 0 :
                        # print(similarity, data['text'])
                        similarity = 0
                else:
                    print("Wrong SFT Type!")
                
                score = str(qid_to_score[question_id])
                similarity_scores[score][question_id] = similarity
                output_type_scores[score][question_id] = output_type

                score_sum += eval(score)
                # print(score, score_sum)

                jsonl_dict ={
                    "part_num:": len(split_parts),
                    "similarity_score": similarity,
                    "output_type": output_type,
                    # "is_true": is_true,
                    "question_id": question_id,
                    "original_output": data['text'],
                    "split_part_lst": split_parts,
                    "original_question": real_question,
                    # QA-SFT
                    "similarity_score_lst": similarity_score_lst,
                    "max_score_sentence_num": max_sentence_num,
                }
                analysis_data.append(jsonl_dict)

    with open(args.analysis_file, "w", encoding="utf-8") as f:
        for entry in analysis_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    sim_high_scores = {}  # >=0.5
    type_high_scores = {}
    sim_low_scores = {}   # <0.5
    type_low_scores = {}
    for score in similarity_scores:
        score_float = float(score)
        sim_scores = similarity_scores[score]
        type_scores = output_type_scores[score]
        if score_float >= 0.5:
            sim_high_scores.update(sim_scores)
            type_high_scores.update(type_scores)
        else:
            sim_low_scores.update(sim_scores)
            type_low_scores.update(type_scores)
    
    sim_high_avg = sum(sim_high_scores.values()) / len(sim_high_scores) if sim_high_scores else 0
    sim_low_avg = sum(sim_low_scores.values()) / len(sim_low_scores) if sim_low_scores else 0
    
    print(f"Average QAA for samples whose score>=0.5: {sim_high_avg * 100:.2f}")
    print(f"Average QAA for samples whose score<0.5: {sim_low_avg * 100:.2f}")

    correct_st_cnt = list(type_high_scores.values()).count(sft_type)
    correct_other_cnt = list(type_high_scores.values()).count('others')
    incorrect_st_cnt = list(type_low_scores.values()).count(sft_type)
    incorrect_other_cnt = list(type_low_scores.values()).count('others')
    print(f"correct_st_cnt:{correct_st_cnt}, correct_other_cnt:{correct_other_cnt}, incorrect_st_cnt:{incorrect_st_cnt}, incorrect_other_cnt:{incorrect_other_cnt}")
    
    print(f"TextVQA ACC:{(score_sum / (st_cnt + other_cnt)) * 100:.2f}")
    print(f"Total: {st_cnt} samples for {sft_type}type. The proportion is {(st_cnt / (st_cnt + other_cnt)) * 100:.2f}")
    print(f"Total: {other_cnt} samples for 'others' type. The proportion is {(other_cnt / (st_cnt + other_cnt)) * 100:.2f}")

    correct_QAA_percent = correct_st_cnt / (correct_st_cnt + correct_other_cnt) if correct_st_cnt + correct_other_cnt else -1
    correct_others_percent = correct_other_cnt / (correct_st_cnt + correct_other_cnt) if correct_st_cnt + correct_other_cnt else -1
    incorrect_QAA_percent = incorrect_st_cnt / (incorrect_st_cnt + incorrect_other_cnt) if incorrect_st_cnt + incorrect_other_cnt else -1
    incorrect_others_percent = incorrect_other_cnt / (incorrect_st_cnt + incorrect_other_cnt) if incorrect_st_cnt + incorrect_other_cnt else -1

    print(f"Correct answer (score>=0.5): {correct_st_cnt} samples for {sft_type} type. The proportion is {correct_QAA_percent * 100:.2f}.")
    print(f"Correct answer (score>=0.5): {correct_other_cnt} samples for 'others' type. The proportion is {correct_others_percent * 100:.2f}.")
    print(f"Incorrect answer (score<0.5): {incorrect_st_cnt} samples for {sft_type} type. The proportion is {incorrect_QAA_percent * 100:.2f}.")
    print(f"Incorrect answer (score<0.5): {incorrect_other_cnt} samples for 'others' type. The proportion is {incorrect_others_percent * 100:.2f}.")

    print(f"Total: Average QAA of all samples: {(sim_high_avg * len(sim_high_scores) + sim_low_avg * len(sim_low_scores)) / 5000 * 100:.2f}")
    print(f"Total: In all samples, The proportion of {sft_type} type is {(correct_QAA_percent * len(sim_high_scores) + incorrect_QAA_percent * len(sim_low_scores)) / 5000 * 100:.2f}")
    print(f"Total: In all samples, The proportion of 'others' type is{(correct_others_percent * len(sim_high_scores) + incorrect_others_percent * len(sim_low_scores)) / 5000 * 100:.2f}")

    output = {
        "similarity_by_score": similarity_scores,
        "high_score_average": sim_high_avg,
        "low_score_average": sim_low_avg,
        "high_scores": sim_high_scores,
        "low_scores": sim_low_scores
    }
    
    if save_file:
        with open(save_file, 'w') as f:
            json.dump(output, f, indent=2)

    # QAA: model_name sft_type task correct incorrect total
    QAA_header = ['model_name', 'sft_type', 'task', 'correct_QAA', 'incorrect_QAA', 'total_QAA', 'acc']
    QAA_dict = {
        'model_name': model_name,
        'sft_type': sft_type,
        'task': 'textvqa',
        'correct_QAA': sim_high_avg * 100,
        'incorrect_QAA': sim_low_avg * 100,
        'total_QAA': (sim_high_avg * len(sim_high_scores) + sim_low_avg * len(sim_low_scores)) / 5000 * 100,
        'acc': (score_sum / (st_cnt + other_cnt)) * 100
    }
    QAA_file_exists = os.path.exists(QAA_CSV_PATH)
    with open(QAA_CSV_PATH, mode='a', newline='', encoding='utf-8') as f:

        writer = csv.DictWriter(f, fieldnames=QAA_header)


        if not QAA_file_exists:
            writer.writeheader()
        
        writer.writerow(QAA_dict)

    # Output_type: model_name sft_type task correct(T/F) incorrect(T/F) total(T/F) 
    output_type_header = ['model_name', 'sft_type', 'task', 'correct_True_output_type', 'correct_False_output_type',
     'incorrect_True_output_type', 'incorrect_False_output_type', 'total_True_output_type', 'total_False_output_type']
    output_type_dict = {
        'model_name': model_name,
        'sft_type': sft_type,
        'task': 'textvqa',
        'correct_True_output_type': correct_QAA_percent * 100,
        'correct_False_output_type': correct_others_percent * 100,
        'incorrect_True_output_type': incorrect_QAA_percent * 100,
        'incorrect_False_output_type': incorrect_others_percent * 100,
        'total_True_output_type': (correct_QAA_percent * len(sim_high_scores) + incorrect_QAA_percent * len(sim_low_scores)) / 5000 * 100,
        'total_False_output_type': (correct_others_percent * len(sim_high_scores) + incorrect_others_percent * len(sim_low_scores)) / 5000 * 100
    }

    output_type_file_exists = os.path.exists(OUTPUT_TYPE_CSV_PATH)

    with open(OUTPUT_TYPE_CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=output_type_header)

        if not output_type_file_exists:
            writer.writeheader()

        writer.writerow(output_type_dict)


def process_gqa(reference_file, prediction_file, inference_file, model_name, split_word='ASSISTANT'):

    def evaluate_predictions(reference_path, prediction_path):
        with open(reference_path, 'r') as f:
            reference_data = json.load(f)
        
        with open(prediction_path, 'r') as f:
            predictions = json.load(f)
        
        pred_dict = {item['questionId']: item['prediction'].strip().lower() for item in predictions}
        
        question_id_dict = {
            'correct': [],
            'incorrect': []
        }
        total = 0

        for qid, ref in reference_data.items():
            if qid in pred_dict:
                total += 1
                gt_answer = ref['answer'].strip().lower()
                pred_answer = pred_dict[qid]
                if gt_answer == pred_answer:
                    question_id_dict['correct'].append(qid)
                else:
                    question_id_dict['incorrect'].append(qid)
        
        accuracy = len(question_id_dict['correct']) / total if total > 0 else 0.0

        print(f"Total evaluated: {total}")
        print(f"Correct: {len(question_id_dict['correct'])}")
        print(f"Incorrect: {len(question_id_dict['incorrect'])}")
        print(f"Accuracy: {accuracy:.2%}")

        return question_id_dict


    sft_type = sft_type_dict[model_name]
    print("sft_type", sft_type)

    question_id_dict = evaluate_predictions(reference_file, prediction_file)


    with open(inference_file, 'r') as f:
        inference_data = [json.loads(line) for line in f]


    correct_distances = []
    incorrect_distances = []

    analysis_data = []
    st_cnt = 0
    other_cnt = 0
    correct_cnt = 0

    correct_st_cnt = 0 
    correct_other_cnt = 0 
    incorrect_st_cnt = 0 
    incorrect_other_cnt = 0 


    inference_dict = {}
    for item in inference_data:

        if item['text'].startswith('USER: \n\n ASSISTANT: QUESTION:'):
            split_parts_text = item['text'][29:]
        elif item['text'].startswith('USER: \n\n ASSISTANT:'):
            split_parts_text = item['text'][19:]
        elif  item['text'].startswith('USER: \n\n'):
            split_parts_text = item['text'][8:]

        elif item['text'].startswith('QUESTION:'):
            split_parts_text = item['text'][9:]
        else:
            split_parts_text = item['text']
        split_parts_text = split_parts_text.replace('<image>', '')
        # split_parts_text = split_parts_text.replace('\n', ' ')
        split_parts_text = split_parts_text.strip()

        real_question = item['prompt']

        begin_with_split_word = False
        if sft_type in ['RQA', 'RQRA'] and split_parts_text.startswith('assistant\n'):
            begin_with_split_word = True
            split_parts_text = split_parts_text[10:]

        if sft_type not in ['R-QA',  'QA', 'RQA']:
            model_split_word = split_word_dict[model_name]
        else:
            model_split_word = split_word
        split_parts = split_parts_text.split(f'{model_split_word}')

        if sft_type in ['R-QA',  'QA', 'RQA']:
            if sft_type in ['R-QA', 'QA'] and len(split_parts) == 1:
                pred_question = split_parts[0].replace('\n', ' ').strip()
            elif sft_type in ['RQA'] and begin_with_split_word and len(split_parts) == 1:
                pred_question = split_parts[0].replace('\n', ' ').strip()

            else:
                pred_question = split_parts_text.replace('\n', ' ').strip()

            split_sentences_lst = split_sentences(pred_question)
            
            input_sentences_lst = []
            current = []
            similarity_score_lst = []
            max_similarity_score = 0
            max_sentence_num = 0
            for j, s in enumerate(split_sentences_lst):
                current.append(s)
                current_sentence = " ".join(current)
                input_sentences_lst.append(current_sentence)
                similarity_score = compute_edit_distance(real_question, current_sentence)
                # modified
                if similarity_score < 0 :
                    # print(similarity_score, question_list[i])
                    similarity_score = 0
                if max_similarity_score < similarity_score:
                    max_similarity_score = similarity_score
                    max_sentence_num = j + 1

                similarity_score_lst.append(similarity_score)
            
            similarity = max_similarity_score

            sentence_num = len(split_sentences_lst)
            if len(split_parts) == 1 and sentence_num > max_sentence_num and max_similarity_score > 0: # Q+A
                if (sft_type in ['RQA'] and begin_with_split_word) or sft_type in ['R-QA',  'QA', ]:
                    output_type = sft_type
                    st_cnt += 1
                else:
                    output_type = 'others'
                    other_cnt += 1                        
            else:
                output_type = 'others'
                other_cnt += 1

        elif sft_type in ['R-QRA', 'QRA', 'RQRA']:
            similarity_score_lst = None
            max_sentence_num = None
    
            if sft_type in ['R-QRA', 'QRA'] and len(split_parts) == 2:
                pred_question = split_parts[0].replace('\n', ' ').strip()
                st_cnt += 1 
                output_type = sft_type
            elif sft_type in ['RQRA'] and begin_with_split_word and len(split_parts) == 2:
                pred_question = split_parts[0].replace('\n', ' ').strip()
                st_cnt += 1 
                output_type = sft_type                    
            else:
                pred_question = split_parts_text.replace('\n', ' ').strip()
                other_cnt += 1
                output_type = 'others'
            
            # print([real_question, pred_question])
            similarity = compute_edit_distance(real_question, pred_question)

            # modified
            if similarity < 0 :
                # print(similarity, data['text'])
                similarity = 0

        else:
            print("Wrong SFT Type!")


        if item['question_id'] in question_id_dict['correct']:
            is_true = True
            correct_distances.append(similarity)
            correct_cnt += 1
            if output_type != 'others':
                correct_st_cnt += 1
            else:
                correct_other_cnt += 1
        elif item['question_id'] in question_id_dict['incorrect']:
            is_true = False
            incorrect_distances.append(similarity)
            if output_type != 'others':
                incorrect_st_cnt += 1
            else:
                incorrect_other_cnt += 1
        else:
            continue

        jsonl_dict ={
            "part_num:": len(split_parts),
            "similarity_score": similarity,
            "output_type": output_type,
            "is_true": is_true,
            "question_id": item['question_id'],
            "original_output": item['text'],
            "split_part_lst": split_parts,
            "original_question": real_question,
            # QA-SFT 
            "similarity_score_lst": similarity_score_lst,
            "max_score_sentence_num": max_sentence_num,
        }
        # print(jsonl_dict)
        analysis_data.append(jsonl_dict)

    with open(args.analysis_file, "w", encoding="utf-8") as f:
        for entry in analysis_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    avg_correct_distance = sum(correct_distances) / len(correct_distances) if correct_distances else 0
    avg_incorrect_distance = sum(incorrect_distances) / len(incorrect_distances) if incorrect_distances else 0

    print(f"Average edit distance for correct predictions (QAA): {avg_correct_distance * 100:.2f}")
    print(f"Average edit distance for incorrect predictions (QAA): {avg_incorrect_distance * 100:.2f}")

    final_acc = correct_cnt / (st_cnt + other_cnt)
    print(f"GQA ACC: {final_acc * 100:.2f}")
    print(f"Total: {st_cnt} samples for {sft_type} type.  The proportion is {(st_cnt / (st_cnt + other_cnt)) * 100:.2f}")
    print(f"Total: {other_cnt} samples for 'others' type.  The proportion is {(other_cnt / (st_cnt + other_cnt)) * 100:.2f}")

    correct_QAA_percent = correct_st_cnt / (correct_st_cnt + correct_other_cnt) if correct_st_cnt + correct_other_cnt else -1
    correct_others_percent = correct_other_cnt / (correct_st_cnt + correct_other_cnt) if correct_st_cnt + correct_other_cnt else -1
    incorrect_QAA_percent = incorrect_st_cnt / (incorrect_st_cnt + incorrect_other_cnt) if incorrect_st_cnt + incorrect_other_cnt else -1
    incorrect_others_percent = incorrect_other_cnt / (incorrect_st_cnt + incorrect_other_cnt) if incorrect_st_cnt + incorrect_other_cnt else -1

    print(f"Correct answer: {correct_st_cnt} samples for {sft_type} type. The proportion is {correct_QAA_percent * 100:.2f}.")
    print(f"Correct answer: {correct_other_cnt} samples for 'others' type. The proportion is {correct_others_percent * 100:.2f}.")
    print(f"Incorrect answer: {incorrect_st_cnt} samples for {sft_type} type. The proportion is {incorrect_QAA_percent * 100:.2f}.")
    print(f"Incorrect answer: {incorrect_other_cnt} samples for 'others' type. The proportion is {incorrect_others_percent * 100:.2f}.")

    print(f"Total: Average QAA of all samples: {(avg_correct_distance * final_acc + avg_incorrect_distance * (1 - final_acc)) * 100:.2f}")
    print(f"Total: In all samples, The proportion of {sft_type} type is {(correct_QAA_percent * final_acc + incorrect_QAA_percent * (1 - final_acc)) * 100:.2f}")
    print(f"Total: In all samples, The proportion of 'others' type is {(correct_others_percent * final_acc + incorrect_others_percent * (1 - final_acc)) * 100:.2f}")

    # QAA: model_name sft_type task correct incorrect total
    QAA_header = ['model_name', 'sft_type', 'task', 'correct_QAA', 'incorrect_QAA', 'total_QAA', 'acc']
    QAA_dict = {
        'model_name': model_name,
        'sft_type': sft_type,
        'task': 'gqa',
        'correct_QAA': avg_correct_distance * 100,
        'incorrect_QAA': avg_incorrect_distance * 100,
        'total_QAA': (avg_correct_distance * final_acc + avg_incorrect_distance * (1 - final_acc)) * 100,
        'acc': final_acc * 100
    }
    QAA_file_exists = os.path.exists(QAA_CSV_PATH)
    with open(QAA_CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=QAA_header)

        if not QAA_file_exists:
            writer.writeheader()
        
        writer.writerow(QAA_dict)

    # Output_type: model_name sft_type task correct(T/F) incorrect(T/F) total(T/F) 
    output_type_header = ['model_name', 'sft_type', 'task', 'correct_True_output_type', 'correct_False_output_type',
     'incorrect_True_output_type', 'incorrect_False_output_type', 'total_True_output_type', 'total_False_output_type']
    output_type_dict = {
        'model_name': model_name,
        'sft_type': sft_type,
        'task': 'gqa',
        'correct_True_output_type': correct_QAA_percent * 100,
        'correct_False_output_type': correct_others_percent * 100,
        'incorrect_True_output_type': incorrect_QAA_percent * 100,
        'incorrect_False_output_type': incorrect_others_percent * 100,
        'total_True_output_type': (correct_QAA_percent * final_acc + incorrect_QAA_percent * (1 - final_acc)) * 100,
        'total_False_output_type': (correct_others_percent * final_acc + incorrect_others_percent * (1 - final_acc)) * 100
    }
    output_type_file_exists = os.path.exists(OUTPUT_TYPE_CSV_PATH)

    with open(OUTPUT_TYPE_CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=output_type_header)

        if not output_type_file_exists:
            writer.writeheader()
        
        writer.writerow(output_type_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help='task name for analysis')
    parser.add_argument("--split_word", type=str, default="ASSISTANT", help='split word')
    # for SQA
    parser.add_argument("--sqa_resoning_json", type=str, help='SQA original reasoning result path')
    parser.add_argument("--sqa_prediction_json", type=str, help='SQA prediction result path')
    parser.add_argument("--sqa_save_file", type=str, help='SQA analysis results save path')
    # for TextVQA
    parser.add_argument("--textvqa_id_to_score_json", type=str, help='textvqa reasoning id to score json path')
    parser.add_argument("--textvqa_prediction_json", type=str, help='textvqa prediction result path')
    parser.add_argument("--textvqa_save_file", type=str, help='textvqa analysis results save path')
    parser.add_argument("--question_file", type=str)

    # added
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    parser.add_argument("--filter_answer", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--original_benchmark", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--model_type", type=str, default='zero-shot')

    # for GQA
    parser.add_argument("--gqa_reference_json", type=str, help='GQA annotation json')
    parser.add_argument("--gqa_reasoning_json", type=str, help='GQA original reasoning result path')
    parser.add_argument("--gqa_prediction_json", type=str, help='SQA prediction result path')

    parser.add_argument("--analysis_file", type=str)
    parser.add_argument("--model_name", type=str)

    args = parser.parse_args()

    if args.task == 'scienceqa':
        print("################ SQA ###############")
        process_sqa(args.sqa_prediction_json, args.sqa_resoning_json, args.sqa_save_file, args.model_name, args.split_word)
    elif args.task == 'textvqa':
        print("################ TextVQA ###############")

        # eval_textvqa.py
        if args.result_file is not None:
            final_acc = eval_single(args.annotation_file, args.result_file, args)

        if args.result_dir is not None:
            for result_file in sorted(os.listdir(args.result_dir)):
                if not result_file.endswith('.jsonl'):
                    print(f'Skipping {result_file}')
                    continue
                eval_single(args.annotation_file, os.path.join(args.result_dir, result_file), args)

        process_textvqa(args.textvqa_id_to_score_json, args.textvqa_prediction_json, args.textvqa_save_file, final_acc, args.model_name, args.split_word)
    else:
        print("################ GQA ###############")
        process_gqa(args.gqa_reference_json, args.gqa_prediction_json, args.gqa_reasoning_json, args.model_name, args.split_word)

