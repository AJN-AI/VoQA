import os
import json
import csv
import argparse
from process_answer import extract_answer, str2bool, split_sentences
from eval_others import sft_type_dict, split_word_dict, QAA_CSV_PATH, OUTPUT_TYPE_CSV_PATH

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


def eval_pope(answers, label_file, args):
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]
    question_list = [json.loads(q)['text'] + " Answer the question using a single word or phrase." for q in open(label_file, 'r')]

    pred_question_list = []

    id_lst = []
    parts_lst = []
    for answer in answers:            

        if answer['text'].startswith('USER: \n\n ASSISTANT: QUESTION:'):
            split_parts_text = answer['text'][29:]
        elif answer['text'].startswith('USER: \n\n ASSISTANT:'):
            split_parts_text = answer['text'][19:]
        elif answer['text'].startswith('USER: \n\n'):
            split_parts_text = answer['text'][8:]

        elif answer['text'].startswith('QUESTION:'):
            split_parts_text = answer['text'][9:]
        else:
            split_parts_text = answer['text']
        split_parts_text = split_parts_text.replace('<image>', '')
        # split_parts_text = split_parts_text.replace('\n', ' ')
        split_parts_text = split_parts_text.strip()

        begin_with_split_word = False
        if sft_type in ['RQA', 'RQRA'] and split_parts_text.startswith('assistant\n'):
            begin_with_split_word = True
            split_parts_text = split_parts_text[10:]
        
        if sft_type not in ['R-QA', 'QA', 'RQA']:
            model_split_word = split_word_dict[args.model_name]
        else:
            model_split_word = args.split_word
        split_parts = split_parts_text.split(f'{model_split_word}')

        if sft_type in ['R-QA', 'QA', 'RQA']:
            if sft_type in ['R-QA', 'QA'] and len(split_parts) == 1:
                pred_question = split_parts[0].replace('\n', ' ').strip()
            elif sft_type in ['RQA'] and begin_with_split_word and len(split_parts) == 1:
                pred_question = split_parts[0].replace('\n', ' ').strip()
            else:
                pred_question = split_parts_text.replace('\n', ' ').strip()

        elif sft_type in ['R-QRA', 'QRA', 'RQRA']:
            if sft_type in ['R-QRA', 'QRA'] and len(split_parts) == 2:
                pred_question = split_parts[0].replace('\n', ' ').strip()
            elif sft_type in ['RQRA'] and begin_with_split_word and len(split_parts) == 2:
                pred_question = split_parts[0].replace('\n', ' ').strip()     
            else:
                pred_question = split_parts_text.replace('\n', ' ').strip()    

        pred_question_list.append(pred_question)
        parts_lst.append(split_parts)
        id_lst.append(answer['question_id'])
        
        text = extract_answer(answer['text'], args.filter_answer, args.split_word, 'pope', args.model_type)

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    TP_sim, TN_sim, FP_sim, FN_sim = 0, 0, 0, 0

    # 初始化信息
    analysis_data = []
    standard_cnt = 0
    other_cnt = 0
    TP_st, TN_st, FP_st, FN_st = 0, 0, 0, 0
    TP_other, TN_other, FP_other, FN_other = 0, 0, 0, 0

    for i, (pred, label, id, split_parts) in enumerate(zip(pred_list, label_list, id_lst, parts_lst)):
        # similarity_score = compute_edit_distance(question_list[i].replace('<image>', '').replace('\n', ' ').strip(), pred_question_list[i])

        pred_question = pred_question_list[i]
        real_question = question_list[i].replace('<image>', '').replace('\n', ' ').strip()

        if sft_type in ['T-QA', 'T-QnA', 'QA', 'QnA', 'TQA', 'TQnA']:

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
                if (sft_type in ['TQA', 'TQnA'] and begin_with_split_word) or sft_type in ['T-QA', 'T-QnA', 'QA', 'QnA']:
                    output_type = sft_type
                    standard_cnt += 1
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
                # pred_question = split_parts[0].replace('\n', ' ').strip()
                standard_cnt += 1 
                output_type = sft_type
            elif sft_type in ['RQRA'] and begin_with_split_word and len(split_parts) == 2:
                # pred_question = split_parts[0].replace('\n', ' ').strip()
                standard_cnt += 1 
                output_type = sft_type                    
            else:
                # pred_question = split_parts_text.replace('\n', ' ').strip()
                other_cnt += 1
                output_type = 'others'
            
            # print([real_question, pred_question])
            similarity = compute_edit_distance(real_question, pred_question)

            # modified
            if similarity < 0 :
                # print(similarity, data['text'])
                similarity = 0
        else:
            print("Wrong SFT type!")

        if pred == pos and label == pos:
            TP += 1
            TP_sim += similarity
            is_true = True
            if output_type != 'others':
                TP_st += 1
            else:
                TP_other += 1
        elif pred == pos and label == neg:
            FP += 1
            FP_sim += similarity
            is_true = False
            if output_type != 'others':
                FP_st += 1
            else:
                FP_other += 1
        elif pred == neg and label == neg:
            TN += 1
            TN_sim += similarity
            is_true = True
            if output_type != 'others':
                TN_st += 1
            else:
                TN_other += 1
        elif pred == neg and label == pos:
            FN += 1
            FN_sim += similarity
            is_true = False
            if output_type != 'others':
                FN_st += 1
            else:
                FN_other += 1

        jsonl_dict ={
            "part_num:": len(split_parts),
            "similarity_score": similarity,
            "output_type": output_type,
            "is_true": is_true,
            "question_id": id,
            "original_output": answers[i]['text'],
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

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    correct_sim = (TP_sim + TN_sim) / (TP + TN) if (TP + TN) > 0 else 0
    incorrect_sim = (FP_sim + FN_sim) / (FP + FN) if (FP + FN) > 0 else 0
    
    print('Average QAA for correct answers: {:.3f}'.format(correct_sim))
    print('Average QAA for incorrect answers: {:.3f}'.format(incorrect_sim))


    acc_st = (TP_st + TN_st) / standard_cnt if standard_cnt else -1
    print(f"{sft_type} Type: {standard_cnt}({TP_st + TN_st + FP_st + FN_st}) samples. {TP_st + TN_st} samples are correct, whose ACC is {acc_st}.")
    print(f"{other_cnt} samples for 'others' type, in which {TP_other + TN_other} samples are correct.")

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio) )
    return acc, correct_sim, incorrect_sim, standard_cnt, TP_st + TN_st, acc_st, other_cnt, TP_other + TN_other, FP_st + FN_st, FP_other + FN_other


def cal_pope_acc(acc_dict):
    final_acc = 0
    for category, acc in acc_dict.items():
        if category == 'random':
            final_acc += 2910 * acc
        else:
            final_acc += 3000 * acc
    return final_acc / 8910.0


if __name__ == "__main__":
    print("################ POPE ###############")
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--filter_answer", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--split_word", type=str, default='ASSISTANT')
    parser.add_argument("--model_type", type=str, default='zero-shot')
    parser.add_argument("--analysis_file", type=str)
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()

    sft_type = sft_type_dict[args.model_name]

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]
    acc_dict = {}
    correct_dict = {}
    incorrect_dict = {}
    acc_st_dict = {}
    total_st_cnt = 0
    total_st_correct_cnt = 0
    total_other_cnt = 0
    total_other_correct_cnt = 0
    total_st_incorrect_cnt = 0
    total_other_incorrect_cnt = 0
    for file in os.listdir(args.annotation_dir):
        assert file.startswith('coco_pope_')
        assert file.endswith('.json')
        category = file[10:-5]
        cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
        print('Category: {}, # samples: {}'.format(category, len(cur_answers)))
        acc, correct_sim, incorrect_sim, standard_cnt, standard_correct_cnt, acc_st, other_cnt, other_correct_cnt, st_incorrect_cnt, other_incorrect_cnt = eval_pope(cur_answers, os.path.join(args.annotation_dir, file), args)
        print("====================================")
        acc_dict[category] = acc
        correct_dict[category] = correct_sim
        incorrect_dict[category] = incorrect_sim
        acc_st_dict[category] = acc_st
        total_st_cnt += standard_cnt
        total_st_correct_cnt += standard_correct_cnt
        total_other_cnt += other_cnt
        total_other_correct_cnt += other_correct_cnt
        total_st_incorrect_cnt += st_incorrect_cnt
        total_other_incorrect_cnt += other_incorrect_cnt
    
    # cal final acc
    final_acc = cal_pope_acc(acc_dict)
    print(f"Weighted average accuracy:", final_acc)
    final_correct = cal_pope_acc(correct_dict)
    final_incorrect = cal_pope_acc(incorrect_dict)
    print('Weighted average QAA for correct answers: {:.2f}'.format(final_correct * 100))
    print('Weighted average QAA for incorrect answers: {:.2f}'.format(final_incorrect * 100))

    final_st_acc = cal_pope_acc(acc_st_dict)
    print(f"Weighted average {sft_type} format accuracy:", final_st_acc)
    print(f"Total: {total_st_cnt} samples for {sft_type} type. The proportion is {(total_st_cnt / (total_st_cnt + total_other_cnt)) * 100:.2f}.")
    print(f"Total: {total_other_cnt} samples for 'others' type. The proportion is {(total_other_cnt / (total_st_cnt + total_other_cnt)) * 100:.2f}.")

    correct_QAA_percent = total_st_correct_cnt / (total_st_correct_cnt + total_other_correct_cnt)
    correct_others_percent = total_other_correct_cnt / (total_st_correct_cnt + total_other_correct_cnt)
    incorrect_QAA_percent = total_st_incorrect_cnt / (total_st_incorrect_cnt + total_other_incorrect_cnt)
    incorrect_others_percent = total_other_incorrect_cnt / (total_st_incorrect_cnt + total_other_incorrect_cnt)
    print(f"Correct answers: {total_st_correct_cnt} samples for {sft_type} type. The proportion is {correct_QAA_percent * 100:.2f}.")
    print(f"Correct answers: {total_other_correct_cnt} samples for 'others' type. The proportion is {correct_others_percent * 100:.2f}.")
    print(f"Incorrect answers: {total_st_incorrect_cnt} samples for {sft_type} type. The proportion is {incorrect_QAA_percent * 100:.2f}.")
    print(f"Incorrect answers: {total_other_incorrect_cnt} samples for 'others' type. The proportion is {incorrect_others_percent * 100:.2f}.")

    print(f"Total: Average QAA for all samples: {(final_correct * final_acc + final_incorrect * (1 - final_acc)) * 100:.2f}")
    print(f"Total: In all samples, The proportion of {sft_type} type is {(correct_QAA_percent * final_acc + incorrect_QAA_percent * (1 - final_acc)) * 100:.2f}")
    print(f"Total: In all samples, The proportion of 'others' type is {(correct_others_percent * final_acc + incorrect_others_percent * (1 - final_acc)) * 100:.2f}")

    # QRA: model_name sft_type task correct incorrect total
    QAA_header = ['model_name', 'sft_type', 'task', 'correct_QAA', 'incorrect_QAA', 'total_QAA', 'acc']
    QAA_dict = {
        'model_name': args.model_name,
        'sft_type': sft_type,
        'task': 'pope',
        'correct_QAA': final_correct * 100,
        'incorrect_QAA': final_incorrect * 100,
        'total_QAA': (final_correct * final_acc + final_incorrect * (1 - final_acc)) * 100,
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
        'model_name': args.model_name,
        'sft_type': sft_type,
        'task': 'pope',
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