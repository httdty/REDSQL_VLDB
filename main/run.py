# -*- coding: utf-8 -*-
# @Time    : 2023/5/6 19:44
# @Author  :
# @Email   :  
# @File    : run.py
# @Software: PyCharm
import json
import time

from tqdm import tqdm

from main.args import *
from bug_fix.post_fix import BugFix
from eval.spider_evaluator import EvaluateTool as SpiderEvaluateTool
from eval.bird_evaluator import EvaluateTool as BirdEvaluateTool

from llms import model_init
from bug_fix.consistency import consistency
from main.load_data import data, load_gold_data
from main.utils import clean_output, load_data, toy, package_sql
from red.red import RED


def main() -> None:
    args = parse_args()
    log_args(args)
    exp_name = output_name(args)

    # Exp records
    dev_ori = None
    if args.stage == 'dev':
        dev_ori = load_gold_data(args)
        if args.toy:
            dev_ori = toy(dev_ori)
        if 'bird' in args.db_dir:
            evaluator = BirdEvaluateTool(iterate_num=2, meta_time_out=30)
            evaluator.register_golds(dev_ori, args.db_dir)
        else:
            evaluator = SpiderEvaluateTool()
            evaluator.register_golds(dev_ori, args.db_dir, args.sample_db_dir)
    else:
        evaluator = None

    # Load dev data
    dev_data = data(args)
    if args.toy:
        dev_data = toy(dev_data)
    dev_data = [dev_data[i:i + args.batch_size] for i in range(0, len(dev_data), args.batch_size)]

    # Init model
    model = model_init(args.model_name, **model_args(args))
    
    # Init prompter
    red = RED(args.train_table_file, args.table_file, args.db_dir, args.bug_only)


    # Bug fix
    bug_fixer = None
    if args.bug_fix:
        bug_fixer_dev_ori = load_data(args.dev_file)
        if args.toy:
            bug_fixer_dev_ori = toy(bug_fixer_dev_ori)
        bug_fixer = BugFix(args.db_dir, args.table_file, bug_fixer_dev_ori)

    out = open(os.path.join(args.output_dir, f"{exp_name}.txt"), 'w')
    out_top = open(os.path.join(args.output_dir, f"{exp_name}_top{args.consistency_num}.txt"), 'w')
    out_log = []
    em = []
    ex = []
    ts = []


    idx = 0
    preds = []
    db_ids = []
    batches = tqdm(dev_data, ncols=168)
    # time.sleep(7200)
    for batch_ins in batches:
        batch_prompt = red.refine(batch_ins)
        batch_raw_output = model.infer(batch_prompt, n=args.consistency_num)
        for i, raw_output in enumerate(batch_raw_output):
            # Out put clean
            results = clean_output(batch_ins[i]['pred'], raw_output)
            for res in results:
                out_top.write(res + "\n")
            out_top.write("\n")
            if bug_fixer:
                for res_idx in range(len(results)):
                    results[res_idx] = bug_fixer.online_fix(idx, results[res_idx])
            if len(results) > 1:
                result = consistency(results, batch_ins[i]['db_id'], args.db_dir)
            else:
                result = results[0]

            # Eval
            if evaluator:
                score = evaluator.evaluate_one(idx=idx, prediction=result)
            else:
                score = {
                    'exact_match': 1,
                    'exec_match': 1,
                    'test_suite_match': 1,
                }
            em.append(score['exact_match'])
            ex.append(score['exec_match'])
            ts.append(score['test_suite_match'])

            # Log info
            if batch_prompt[i]:
                logger.info(batch_prompt[i])
                logger.info(raw_output[0])
                logger.info(result)
                # if args.stage == 'dev':
                if dev_ori:
                    logger.info("Gold: " + dev_ori[idx]['query'])
                logger.info(f"{ex[-1] != 0} \t{batch_ins[i]['db_id']}")
                print()

            idx += 1
            if idx % 77 == 0:
                print()
            if model.count == 0:
                batches.desc = f"EM: {sum(em) / idx * 100:.2f}%   " \
                               f"EX: {sum(ex) / idx * 100:.2f}%   " \
                               f"TS: {sum(ts) / idx * 100:.2f}%   " \
                               f"NUM: {model.count} "
            else:
                batches.desc = f"EM: {sum(em) / idx * 100:.2f}%   " \
                               f"EX: {sum(ex) / idx * 100:.2f}%   " \
                               f"TS: {sum(ts) / idx * 100:.2f}%   " \
                               f"IN: {model.prompt_length / model.count:.1f}   " \
                               f"OUT: {model.completion_length / model.count:.1f}   " \
                               f"NUM: {model.count} "

            # File log
            out_log.append({
                "prompt": batch_prompt[i],
                "result": result,
                "raw_result": raw_output,
                "mark": score
            })
            result = result.replace("\n", " ")
            preds.append(result)
            db_ids.append(batch_ins[i]['db_id'])
            out.write(f"{str(result)}\n")
            out.flush()
            out_top.flush()

    res = package_sql(preds, db_ids)
    with open("./predict.json", 'w') as f:
        json.dump(res, f, indent=2)

    # Stat info
    if evaluator:
        evaluator.print_score()
    if model.count != 0:
        logger.info(
            f"\nExact match\t{sum(em) / idx * 100:.2f}%"
            f"\nExec match \t{sum(ex) / idx * 100:.2f}%"
            f"\nTest suite \t{sum(ts) / idx * 100:.2f}%"
            f"\nPrompt     \t{model.prompt_length / model.count:.1f}"
            f"\nCompletion \t{model.completion_length / model.count:.1f}"
            f"\nNumber     \t{model.count}"
        )
    logger.info(f"Exp name: {exp_name}")
    logger.info(f"Output dir: {args.output_dir}")
    with open(os.path.join(args.output_dir, f"{exp_name}.json"), 'w') as f:
        json.dump(out_log, f, indent=4)

    if bug_fixer:
        logger.info(f"Fix and pass number: {bug_fixer.fix_pass}")
        logger.info(f"Fix but fail number: {bug_fixer.fix_fail}")
        reasons = '   '.join(list(set(bug_fixer.fail_reason)))
        logger.info(f"Failed reasons: {reasons}")

    logger.info("Prepare file for eval...")
    exp_abs = os.path.abspath(os.path.join(args.output_dir, f"{exp_name}.txt"))
    target_file = os.path.join(os.path.abspath("./"), "predicted_sql.txt")
    os.system(f"cp {exp_abs} {target_file}")
    logger.info("Finished")


if __name__ == "__main__":
    main()
