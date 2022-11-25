import argparse
import torch
import os

from utils import load_file, load_dir, write_to_csv
from metrics import PTM2


def main():
    parser = argparse.ArgumentParser("PT-M2")
    parser.add_argument("--source", type=str, default="source file path")
    parser.add_argument("--reference", type=str, default="reference file path")
    parser.add_argument("--hypothesis", type=str, default="hypothesis file path")
    parser.add_argument("--output", type=str, default="output file path")
    parser.add_argument("--base", choices=["m2", "sentm2", "errant", "senterrant"], default="m2", type=str)
    parser.add_argument("--scorer", choices=["self", "bertscore", "bartscore"],
                        default="self", type=str, help="choose the plm scorer type")
    parser.add_argument("--model_type", type=str, help="choose the plm type", default="bert-base-uncased")
    parser.add_argument("--beta", default=0.5, type=float)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print(args)

    sources = load_file(args.source)
    references = load_dir(args.reference)
    m2_file = f"{args.reference}.m2"

    metric = PTM2(args, corpus=None)

    if args.base == "m2":
        score = metric.compute_m2(m2_file=m2_file, hyp_file=args.hypothesis, sources=sources, references=references)
    elif args.base == "sentm2":
        score = metric.compute_sentm2(m2_file=m2_file, hyp_file=args.hypothesis, sources=sources, references=references)
    elif args.base == "errant":
        score = metric.compute_errant(m2_file=m2_file, hyp_file=args.hypothesis, sources=sources, references=references)
    elif args.base == "senterrant":
        score = metric.compute_senterrant(m2_file=m2_file, hyp_file=args.hypothesis, sources=sources, references=references)

    print(f"base={args.base}, scorer={args.scorer}, model_type={args.model_type}, score={score:.4f}")
    with open(args.output, "w", encoding="utf8") as fw:
        fw.write(f"base={args.base}, scorer={args.scorer}, model_type={args.model_type}, score={score}")


if __name__ == "__main__":
    main()
