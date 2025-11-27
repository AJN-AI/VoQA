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
    parser.add_argument("--textvqa_id_to_score_json", type=str, help='textvqa id_to_score json path')
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


def eval_single(annotation_file, result_file, args):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    annotations = json.load(open(annotation_file))['data']
    annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    for result in results:
        annotation = annotations[(result['question_id'], prompt_processor(result['prompt'], args))]
        pred_list.append({
            "question_id": result['question_id'], 
            "pred_answer": extract_answer(result['text'], args.filter_answer, args.split_word, 'textvqa', args.model_type),
            # "pred_answer": extract_answer(result['text'], args.filter_answer, args.split_word),
            "gt_answers": annotation['answers'],
        })

    evaluator = TextVQAAccuracyEvaluator()
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list, args.textvqa_id_to_score_json)))


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
