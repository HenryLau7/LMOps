import requests
import os
import evaluators
import concurrent.futures
from tqdm import tqdm
import time
import json
import argparse
import scorers
import tasks
import predictors
import optimizers
from utils import log_to_file

def get_task_class(task_name):
    if task_name == 'ethos':
        return tasks.EthosBinaryTask
    elif task_name == 'jailbreak':
        return tasks.JailbreakBinaryTask
    elif task_name == 'liar':
        return tasks.DefaultHFBinaryTask
    elif task_name == 'ar_sarcasm':
        return tasks.DefaultHFBinaryTask
    elif task_name == 'BigBench':
        return tasks.BinaryClassificationTask
    else:
        raise Exception(f'Unsupported task: {task_name}')


def get_evaluator(evaluator):
    if evaluator == 'bf':
        return evaluators.BruteForceEvaluator
    elif evaluator in {'ucb', 'ucb-e'}:
        return evaluators.UCBBanditEvaluator
    elif evaluator in {'sr', 's-sr'}:
        return evaluators.SuccessiveRejectsEvaluator
    elif evaluator == 'sh':
        return evaluators.SuccessiveHalvingEvaluator
    else:
        raise Exception(f'Unsupported evaluator: {evaluator}')



def get_scorer(scorer):
    if scorer == '01':
        return scorers.Cached01Scorer
    elif scorer == 'll':
        return scorers.CachedLogLikelihoodScorer
    else:
        raise Exception(f'Unsupported scorer: {scorer}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='liar')
    parser.add_argument('--data_dir', default='data/liar')
    parser.add_argument('--prompts', default='prompts/liar.md')
    # parser.add_argument('--config', default='default.json')
    parser.add_argument('--out', default='test_out.txt')
    parser.add_argument('--max_threads', default=32, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)

    parser.add_argument('--optimizer', default='nl-gradient')
    parser.add_argument('--rounds', default=20, type=int)
    parser.add_argument('--beam_size', default=8, type=int)
    parser.add_argument('--n_test_exs', default=-1, type=int)
    parser.add_argument('--eval_llm', default='Mistral')

    parser.add_argument('--minibatch_size', default=40, type=int)
    parser.add_argument('--n_gradients', default=4, type=int)
    parser.add_argument('--errors_per_gradient', default=5, type=int)
    parser.add_argument('--gradients_per_error', default=1, type=int)
    parser.add_argument('--steps_per_gradient', default=1, type=int)
    parser.add_argument('--mc_samples_per_step', default=2, type=int)
    parser.add_argument('--max_expansion_factor', default=8, type=int)

    parser.add_argument('--engine', default="chatgpt", type=str)

    parser.add_argument('--evaluator', default="bf", type=str)
    parser.add_argument('--scorer', default="01", type=str)
    parser.add_argument('--eval_rounds', default=8, type=int)
    parser.add_argument('--eval_prompts_per_round', default=8, type=int)
    # calculated by s-sr and sr
    parser.add_argument('--samples_per_eval', default=32, type=int)
    parser.add_argument('--c', default=1.0, type=float, help='exploration param for UCB. higher = more exploration')
    parser.add_argument('--knn_k', default=2, type=int)
    parser.add_argument('--knn_t', default=0.993, type=float)
    parser.add_argument('--reject_on_errors', action='store_true') 
    parser.add_argument('--gpu_id', default='2')
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    config = vars(args)

    config['eval_budget'] = config['samples_per_eval'] * config['eval_rounds'] * config['eval_prompts_per_round']

    vllm_model_path={
        "Mistral": "../../Mistral-7B-v0.1",
        "Phi3": "../../Phi-3-mini-4k-instruct",
        "Llama3.1": "../../Meta-Llama-3.1-8B",
        "Llama3.0-Instruct": "../../Meta-Llama-3-8B-Instruct",
    }

    import os
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_folder = os.path.join('./result/', args.task, args.eval_llm, timestamp)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    model_call_out_file =  output_folder + '/api_call.txt'
    os.environ["GPT_CALL_OUT_FILE_NAME"] = model_call_out_file

    args.out = os.path.join(output_folder, args.out)
    if not os.path.exists(os.path.dirname(args.out)):
        os.makedirs(os.path.dirname(args.out))
    os.environ["LOG_FILE"] = args.out
    
    if args.task.startswith('BigBench'):
        sub_task = args.task.split('-')[1]
        task = tasks.BigBenchTask(args.data_dir, sub_task, args.max_threads)
    else:
        task = get_task_class(args.task)(args.data_dir, args.max_threads)

    scorer = get_scorer(args.scorer)()
    evaluator = get_evaluator(args.evaluator)(config)
    bf_eval = get_evaluator('bf')(config)
    # gpt4 = predictors.BinaryPredictor(config)

    vllm = predictors.Vllm_predictor(
        model_path = vllm_model_path[args.eval_llm],
        max_tokens=3,
        stop=None,
        repetition_penalty=1.0,
        top_p=0.1,
        temperature=0,
    )

    optimizer = optimizers.ProTeGi(
        config, evaluator, scorer, args.max_threads, bf_eval)

    train_exs = task.get_train_examples()
    test_exs = task.get_test_examples()
    val_exs = task.get_val_examples()


    if os.path.exists(args.out):
        os.remove(args.out)

    print(config)

    with open(args.out, 'a') as outf:
        outf.write(json.dumps(config) + '\n')

    log_to_file(args.out, f"Training examples: {len(train_exs)}\n")
    log_to_file(args.out, f"Validation examples: {len(val_exs)}\n")
    log_to_file(args.out, f"Test examples: {len(test_exs)}\n")
    

    if args.task.startswith('BigBench'):
        candidates = [open(f"/home/aiscuser/LMOps/prompt_optimization/prompts/BigBench/{sub_task}_2.md").read()]
    else:
        candidates = [open(fp.strip()).read() for fp in args.prompts.split(',')]


    for round in tqdm(range(config['rounds'] + 1)):
        print("STARTING ROUND ", round)
        start = time.time()

        # expand candidates
        if round > 0:
            log_to_file(args.out, f"\n======= START EXPANDING CANDIDATES =======\n")
            candidates = optimizer.expand_candidates(candidates, task, vllm, train_exs)

        # score candidates
        log_to_file(args.out, f"\n======= START SCORE CANDIDATES =======\n")
        scores = optimizer.score_candidates(candidates, task, vllm, val_exs)
        [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), reverse=True)))

        # select candidates
        candidates = candidates[:config['beam_size']]
        scores = scores[:config['beam_size']]

        log_to_file(args.out, f"\n======= START TEST =======\n")
        metrics = []
        for candidate, score in zip(candidates, scores):
            f1, texts, labels, preds = task.evaluate(vllm, candidate, test_exs, n=args.n_test_exs)
            for text, label, pred in zip(texts, labels, preds):
                log_to_file(args.out, f"== Prompt: {text}\n== Label: {task.stringify_prediction(label)}\n== Prediction: {task.stringify_prediction(pred)}== Score: {label==pred}\n")
            metrics.append(f1)

        # record candidates, estimated scores, and true scores
        with open(args.out, 'a') as outf:
            outf.write(f"======== ROUND {round}\n")
            outf.write(f'TIME: {time.time() - start}\n')
            outf.write(f'CANDIDATES: \n{candidates}\n')
            outf.write(f'SCORES: {scores}\n')
        with open(args.out, 'a') as outf:  
            outf.write(f'TEST SCORES: {metrics}\n')
            print(f'TEST SCORES: {metrics}\n')

    print("DONE!")
