import os
import argparse
import json
import re
from tqdm import tqdm
from process_answer import extract_answer, str2bool
from m4c_evaluator import EvalAIAnswerProcessor, TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    parser.add_argument("--filter_answer", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--original_benchmark", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--split_word", type=str, default='ASSISTANT')
    parser.add_argument("--model_type", type=str, default='zero-shot')
    parser.add_argument("--question_file", type=str)
    return parser.parse_args()


def prompt_processor(prompt, args):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    # elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 4:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    # elif len(prompt.split('\n')) == 2:
    elif len(prompt.split('\n')) == 3:
        question = prompt.split('\n')[0]
    # for original benchmark
    elif args.original_benchmark and len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


def load_question_map(jsonl_path):
    question_map = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            qid = entry.get("question_id")
            text = entry.get("text")
            if qid and text:
                question_map[qid] = text.split('\n')[0]
    print("dict length:", len(question_map))
    return question_map


def eval_single(annotation_file, result_file, args):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    annotations = json.load(open(annotation_file))['data']
    annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    qid_to_text = load_question_map(args.question_file)

    for qid, text in list(qid_to_text.items())[:3]:
        print(qid, "=>", text)

    pred_list = []
    for result in results:
        # annotation = annotations[(result['question_id'], prompt_processor(result['prompt'], args))]
        annotation = annotations[(result['question_id'], qid_to_text[result['question_id']].lower())]
        
        pred_answer = extract_answer(result['text'], args.filter_answer, args.split_word, 'textvqa', args.model_type)

        pred_list.append({
            "pred_answer": pred_answer,
            # "pred_answer": extract_answer(result['text'], args.filter_answer, args.split_word),
            "gt_answers": annotation['answers'],
        })
        print([result['question_id'], pred_answer, result['text']])

    evaluator = TextVQAAccuracyEvaluator()
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file, args)

    if args.result_dir is not None:
        for result_file in sorted(os.listdir(args.result_dir)):
            if not result_file.endswith('.jsonl'):
                print(f'Skipping {result_file}')
                continue
            eval_single(args.annotation_file, os.path.join(args.result_dir, result_file), args)
