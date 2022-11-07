import logging
import argparse
import os

from simpletransformers.seq2seq import Seq2SeqModel
from data_reader.data_reader import read_data_source_target


def main():
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
        help="The input data dir. Should contain the source and target files for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
        help="Model type, currently only support seq2seq")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models")

    # Other parameters
    parser.add_argument("--fp16", action="store_true", help="whether use half-precision training")
    parser.add_argument("--do_train", action="store_true", help="Whether run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether run eval on the valid set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction on the test set.")
    parser.add_argument("--init_model_weights", action="store_true", help="Whether to initialize the model weights")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                help="Whether to overwrite on the existing output dir")
    parser.add_argument("--use_multiprocessed_decoding", action="store_true",
                        help="Whether to use multiprocess when decoding")
    parser.add_argument("--save_model_every_epoch", action="store_true",
                        help="Whether to save model every epoch during training")
    parser.add_argument("--predict_during_training", action="store_true",
                        help="Whether to predict after each checkpoint-saving during training")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Whether to evaluate after each checkpoint-saving during training")
    parser.add_argument(
        "--output_dir",
        default='output_dir/', type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--save_step",
        default=0, type=int,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=16, type=int,
        help="Size of each train batch",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=16, type=int,
        help="Size of each eval/predict batch",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=1, type=int,
        help="gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        default=4e-5, type=float,
        help="learning rate",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=100, type=int,
        help="Number of train epochs",
    )
    parser.add_argument(
        "--max_seq_length",
        default=None, type=int,
        help="Max input seq length",
    )
    parser.add_argument(
        "--max_length",
        default=None, type=int,
        help="Max output seq length",
    )
    parser.add_argument(
        "--prediction_dir",
        default=None, type=str,
        help="The output directory where the predictions results will be written.",
    )
    parser.add_argument(
        "--prediction_suffix",
        default=None, type=str,
        help=" The supplementary suffix of prediction results name.",
    )
    parser.add_argument(
        "--mask_ratio",
        default=0.0, type=float,
        help="the proportion of masked words in the source",
    )
    parser.add_argument(
        "--mask_length",
        default="span-poisson", type=str,
        choices=['subword', 'word', 'span-poisson'],
        help="when masking words, the length of mask segments",
    )
    parser.add_argument(
        '--replace_length', default=-1, type=int,
        help='when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)'
    )
    parser.add_argument(
        '--poisson_lambda',
        default=3.0, type=float,
        help='randomly shuffle sentences for this proportion of inputs'
    )
    parser.add_argument(
        '--dataloader_num_workers', default=0, type=int,
        help='the number of cpus used in collecting data in dataloader, '
             'note that if it is large than cpu number, the program may be stuck'
    )
    parser.add_argument(
        '--evaluation_metric', default='qa', type=str,
        help='if pretrain passages, use \'passage\', else use \'qa\''
    )

    ####
    parser.add_argument('--iter_step', default=1, type=int,
                        help='# of steps for iterative generation during prediction')
    parser.add_argument('--manual_seed', default=4, type=int,
                        help='random seed')
    parser.add_argument('--prompt_tuning_mode', action="store_true",
                        help="iterative prompting mode.")
    parser.add_argument('--ptencoder_name', default='roberta-base', type=str,
                        help="name or dir of prompt encoder")
    parser.add_argument('--static_prompt_tuning_mode', action="store_true",
                        help="static prompt tuning mode.")
    parser.add_argument('--use_decoder_pt', action="store_true",
                        help="whether also insert decoder prompts")
    parser.add_argument('--num_encoder_prompt_tokens', default=10, type=int,
                        help='# of prompt tokens prepended to encoder input')
    parser.add_argument('--num_decoder_prompt_tokens', default=10, type=int,
                        help='# of prompt tokens prepended to decoder input')
    parser.add_argument("--predict_on_train", action="store_true", help="Whether predict on the train.")
    parser.add_argument("--predict_on_eval", action="store_true", help="Whether predict on the validation split.")
    parser.add_argument('--train_stopper', action="store_true", help="Whether train the prompt stopper in pt mode.")
    parser.add_argument('--auto_stop', action="store_true",
                        help="Whether use prompt stopper to auto-stop during iterative prediction")
    parser.add_argument('--pt_dir', default=None, type=str,
                        help="directory of prompter, adapter & stopper. Set to model_name_or_path if not provided")

    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int)

    args = parser.parse_args()

    if args.train_stopper or args.auto_stop:
        assert args.prompt_tuning_mode
    if args.prompt_tuning_mode or args.static_prompt_tuning_mode:
        # if in PT mode, the max length needs to be extended to include the prompt tokens
        args.max_seq_length = args.max_seq_length + args.num_encoder_prompt_tokens
        if args.use_decoder_pt:
            args.max_length = args.max_length + args.num_decoder_prompt_tokens
    if args.pt_dir is None:
        args.pt_dir = args.model_name_or_path
    if args.static_prompt_tuning_mode:
        assert not args.prompt_tuning_mode
        assert not args.train_stopper
        assert not args.auto_stop

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if args.do_train or args.predict_on_train:
        train_df = read_data_source_target(args.data_dir + "train.source", args.data_dir + "train.target")
    else:
        train_df = None

    if args.do_eval or args.evaluate_during_training or args.predict_on_eval:
        eval_df = read_data_source_target(args.data_dir + "valid.source", args.data_dir + "valid.target")
    else:
        eval_df = None

    if args.do_predict or args.predict_during_training:
        test_df = read_data_source_target(args.data_dir + "test.source", args.data_dir + "test.target")
    else:
        test_df = None

    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": args.overwrite_output_dir,
        "init_model_weights": args.init_model_weights,
        "max_seq_length": args.max_seq_length,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "save_eval_checkpoints": False,
        "save_model_every_epoch": args.save_model_every_epoch,
        "save_steps": args.save_step,
        "evaluate_during_training": args.evaluate_during_training,
        "evaluate_generated_text": True,
        "evaluate_during_training_verbose": True,
        "predict_during_training": args.predict_during_training,
        "use_multiprocessing": False,
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "manual_seed": args.manual_seed,
        "mask_ratio": args.mask_ratio,
        "mask_length": args.mask_length,
        "replace_length": args.replace_length,
        "poisson_lambda": args.poisson_lambda,
        "fp16": args.fp16,
        "truncation": True,
        "dataloader_num_workers":args.dataloader_num_workers,
        "use_multiprocessed_decoding":args.use_multiprocessed_decoding,
        "evaluation_metric": args.evaluation_metric,
    }

    # Initialize model
    if args.model_type == 'seq2seq':
        model = Seq2SeqModel(
            encoder_decoder_type="bart",
            encoder_decoder_name=args.model_name_or_path,
            args=model_args,
            prompt_tuning_mode=args.prompt_tuning_mode,
            ptencoder_name=args.ptencoder_name,
            static_prompt_tuning_mode=args.static_prompt_tuning_mode,
            use_decoder_pt=args.use_decoder_pt,
            num_encoder_prompt_tokens=args.num_encoder_prompt_tokens,
            num_decoder_prompt_tokens=args.num_decoder_prompt_tokens,
            pt_dir=args.pt_dir,
            local_rank=args.local_rank,
            rank=args.rank,
            gpu=args.gpu,
            world_size=args.world_size,
            dist_url=args.dist_url,
            dist_backend=args.dist_backend,
        )
    else:
        raise ValueError(
            "The {} model is not supported now".format(args.model_type)
        )

    if args.do_train:
        model.train_model(train_data=train_df, eval_data=eval_df, test_data=test_df, output_dir=args.output_dir,
                          train_stopper=args.train_stopper)

    pred_suffix = args.prediction_suffix
    if pred_suffix is None:
        pred_suffix = ""
    if args.do_predict:
        model.predict(pred_data=test_df, output_dir=args.prediction_dir, suffix="test"+pred_suffix,
                      iter_step=args.iter_step, auto_stop=args.auto_stop)
    if args.predict_on_train:
        model.predict(pred_data=train_df, output_dir=args.prediction_dir, suffix="train"+pred_suffix,
                      iter_step=args.iter_step, auto_stop=args.auto_stop)
    if args.predict_on_eval:
        model.predict(pred_data=eval_df, output_dir=args.prediction_dir, suffix="valid"+pred_suffix,
                      iter_step=args.iter_step, auto_stop=args.auto_stop)


if __name__ == '__main__':
    main()
