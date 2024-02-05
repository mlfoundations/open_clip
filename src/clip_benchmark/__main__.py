import argparse
import sys


def get_parser_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    _run_parser = subparsers.add_parser('run', help='Run the CLIP Benchmark')
    _run_parser.add_argument(
        '--dataset',
        dest='datasets',
        type=str,
        default=['cifar10'],
        nargs='+',
        help=(
            'Dataset(s) to use for the benchmark. Can be the name of a dataset, or a '
            "collection name ('vtab', 'vtab+', 'imagenet_robustness', 'retrieval') or "
            'path of a text file where each line is a dataset name'
        ),
    )
    _run_parser.add_argument(
        '--dataset-root',
        default='root',
        type=str,
        help=(
            'Dataset root folder where the datasets are downloaded. Can be in the '
            'form of a template depending on dataset name, e.g. '
            "--dataset-root='datasets/{dataset}'. This is useful "
            'if you evaluate on multiple datasets'
        ),
    )
    _run_parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Dataset split to use, same for all datasets',
    )
    _run_parser.add_argument(
        '--model',
        dest='models',
        type=str,
        default='',
        nargs='+',
        help=(
            'Pre-trained model(s) to use. Can be the full model name where `model` '
            'and `pretrained` are comma separated (e.g., '
            "--pretrained-model='ViT-B-32-quickgelu,laion400m_e32'), a model "
            "collection name ('openai' or 'openclip_base' or 'openclip_multilingual' "
            "or 'openclip_all') or the path to a text file where each line is a model "
            'fullname where model and pretrained are comma separated '
            '(e.g., ViT-B-32-quickgelu,laion400m_e32). Parameters --model and '
            '--pretrained are ignored if --pretrained-model is used'
        ),
    )
    _run_parser.add_argument(
        '--task',
        type=str,
        default='auto',
        choices=[
            'zeroshot_classification',
            'zeroshot_retrieval',
            'linear_probe',
            'captioning',
            'image_caption_selection',
            'auto',
        ],
        help=(
            'Task to evaluate on. With --task=auto, the task is automatically inferred '
            'from the dataset'
        ),
    )
    _run_parser.add_argument(
        '--no-amp',
        action='store_false',
        dest='amp',
        default=True,
        help='Whether to use mixed precision',
    )
    _run_parser.add_argument(
        '--num-workers',
        default=4,
        type=int,
        help='Number of dataloader processes',
    )
    _run_parser.add_argument(
        '--recall-k',
        dest='recall_ks',
        nargs='+',
        default=[5],
        type=int,
        help='For retrieval tasks, select the k for recall@k metrics',
    )
    _run_parser.add_argument(
        '--fewshot-k',
        default=-1,
        type=int,
        help=(
            'For linear probing tasks, select how many shots. When set to -1 the '
            'whole dataset is used'
        ),
    )
    _run_parser.add_argument(
        '--fewshot-epochs',
        default=10,
        type=int,
        help='For linear probing tasks, set the number of training epochs',
    )
    _run_parser.add_argument(
        '--fewshot-lr',
        default=0.1,
        type=float,
        help='For linear probing tasks, set the learning rate',
    )
    _run_parser.add_argument(
        '--linear-probe-train-split',
        dest='linear_probe_train_splits',
        type=str,
        nargs='+',
        default='train',
        help='Dataset(s) train split names, used for linear probing tasks',
    )
    mutually_exclusive = _run_parser.add_mutually_exclusive_group()
    mutually_exclusive.add_argument(
        '--linear-probe-val-split',
        dest='linear_probe_val_splits',
        default=None,
        type=str,
        nargs='+',
        help=(
            'Dataset(s) validation split names, used for linear probing tasks. '
            'Mutually exclusive with --linear-probe-val-proportion'
        ),
    )
    mutually_exclusive.add_argument(
        '--linear-probe-val-proportion',
        dest='linear_probe_val_proportions',
        default=None,
        type=float,
        nargs='+',
        help=(
            'The share of the train datasets that will be used for validation, '
            'if the validation set is not predefined. Mutually exclusive with '
            '--linear-probe-val-split'
        ),
    )
    _run_parser.add_argument(
        '--skip-load',
        action='store_true',
        help=(
            'For linear probing tasks, when everything is cached no need to load model'
        ),
    )
    _run_parser.add_argument(
        '--distributed', action='store_true', help='Run evaluations in parallel'
    )
    _run_parser.add_argument('--seed', default=0, type=int, help='The random seed')
    _run_parser.add_argument(
        '--batch-size', default=64, type=int, help='The batch size'
    )
    _run_parser.add_argument(
        '--normalize', default=True, type=bool, help='Feature normalization'
    )
    _run_parser.add_argument(
        '--model-cache-dir',
        default=None,
        type=str,
        help='The directory where downloaded models are cached',
    )
    _run_parser.add_argument(
        '--feature-root',
        default='features',
        type=str,
        help='The directory where the features are stored',
    )
    _run_parser.add_argument(
        '--annotation-file',
        default=None,
        type=str,
        help=(
            'The text annotation file for retrieval datasets. Only needed '
            'when `--task` is `zeroshot_retrieval`'
        ),
    )
    _run_parser.add_argument(
        '--custom-classname-file',
        default=None,
        type=str,
        help=(
            'Use custom classnames for each dataset. Should be a JSON file where '
            'keys are dataset names and values are lists of classnames'
        ),
    )
    _run_parser.add_argument(
        '--custom-template-file',
        default=None,
        type=str,
        help=(
            'Use custom prompts for each dataset. Should be a JSON file where '
            'keys are dataset names and values are lists of prompts. For instance, '
            "to use CuPL prompts, use --custom-template-file='cupl_prompts.json'"
        ),
    )
    _run_parser.add_argument(
        '--dump-classnames',
        default=False,
        action='store_true',
        help='Dump classnames to the results JSON file',
    )
    _run_parser.add_argument(
        '--dump-templates',
        default=False,
        action='store_true',
        help='Dump templates to the results JSON file',
    )
    _run_parser.add_argument(
        '--language',
        default='en',
        type=str,
        nargs='+',
        help='Language(s) of classnames and prompts to use for zeroshot classification',
    )
    _run_parser.add_argument(
        '--output',
        default='result.json',
        type=str,
        help=(
            'Output file where to dump the metrics. Can be in form of a template, '
            "e.g., --output='{dataset}_{pretrained}_{model}_{language}_{task}.json'"
        ),
    )
    _run_parser.add_argument(
        '--save-clf',
        default=None,
        type=str,
        help='Optionally save the classification layer output by the text tower',
    )
    _run_parser.add_argument(
        '--load-clf',
        dest='load_clfs',
        nargs='+',
        default=[],
        type=str,
        help='Optionally load and average mutliple layer outputs by text towers',
    )
    _run_parser.add_argument(
        '--skip-existing',
        default=False,
        action='store_true',
        help='Whether to skip an evaluation if the output file exists',
    )
    _run_parser.add_argument(
        '--wds-cache-dir',
        default=None,
        type=str,
        help='Optional cache directory for webdatasets only',
    )
    _run_parser.set_defaults(which='run')

    _gather_parser = subparsers.add_parser(
        'gather', help='Gather JSON evaluation files to a table'
    )
    _gather_parser.add_argument(
        'files', type=str, nargs='+', help='Path(s) to JSON result files'
    )
    _gather_parser.add_argument(
        '--output', type=str, default='benchmark.csv', help='CSV output file'
    )
    _gather_parser.set_defaults(which='gather')

    return parser, parser.parse_args()


def _main_gather(args: argparse.Namespace):
    from clip_benchmark.gather import gather_results

    gather_results(fnames=args.files, output=args.output)


def _main_run(args: argparse.Namespace):
    from clip_benchmark.run import run_benchmark

    run_benchmark(
        datasets=args.datasets,
        models=args.models,
        task=args.task,
        output=args.output,
        language=args.language,
        dataset_root=args.dataset_root,
        feature_root=args.feature_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=args.amp,
        distributed=args.distributed,
        skip_existing=args.skip_existing,
        recall_ks=args.recall_ks,
        save_clf=args.save_clf,
        load_clfs=args.load_clfs,
        annotation_file=args.annotation_file,
        custom_template_file=args.custom_template_file,
        custom_classname_file=args.custom_classname_file,
        dump_classnames=args.dump_classnames,
        dump_templates=args.dump_templates,
        model_cache_dir=args.model_cache_dir,
        wds_cache_dir=args.wds_cache_dir,
        normalize=args.normalize,
        split=args.split,
        linear_probe_train_splits=args.linear_probe_train_splits,
        linear_probe_val_splits=args.linear_probe_val_splits,
        linear_probe_val_proportions=args.linear_probe_val_proportions,
        linear_probe_fewshot_k=args.linear_probe_fewshot_k,
        linear_probe_fewshot_lr=args.linear_probe_fewshot_lr,
        linear_probe_fewshot_epochs=args.linear_probe_fewshot_epochs,
    )


def main():
    parser, args = get_parser_args()
    if not hasattr(args, 'which'):
        parser.print_help()
        return
    if args.which == 'run':
        _main_run(args)
    elif args.which == 'gather':
        _main_gather(args)
    else:
        parser.print_help()
        return


if __name__ == '__main__':
    sys.exit(main())
