import json
import logging
import math
import os
import random
import time
import re
import warnings
import builtins
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
import transformers
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BartConfig,
    BartModel,
    BartForConditionalGeneration,
    BartTokenizer,
    BertConfig,
    BertForMaskedLM,
    BertModel,
    BertTokenizer,
    CamembertConfig,
    CamembertModel,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertModel,
    DistilBertTokenizer,
    ElectraConfig,
    ElectraModel,
    ElectraTokenizer,
    EncoderDecoderConfig,
    EncoderDecoderModel,
    LongformerConfig,
    LongformerModel,
    LongformerTokenizer,
    MarianConfig,
    MarianMTModel,
    MarianTokenizer,
    MobileBertConfig,
    MobileBertModel,
    MobileBertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import Seq2SeqArgs
from simpletransformers.seq2seq.seq2seq_utils import Seq2SeqDataset, SimpleSummarizationDataset

from soft_embedding import LEmbedding, SoftEmbedding
from utils import FeedForwardNet

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "auto": (AutoConfig, AutoModel, AutoTokenizer),
    "bart": (BartConfig, BartForConditionalGeneration, BartTokenizer),
    "bert": (BertConfig, BertModel, BertTokenizer),
    "camembert": (CamembertConfig, CamembertModel, CamembertTokenizer),
    "distilbert": (DistilBertConfig, DistilBertModel, DistilBertTokenizer),
    "electra": (ElectraConfig, ElectraModel, ElectraTokenizer),
    "longformer": (LongformerConfig, LongformerModel, LongformerTokenizer),
    "mobilebert": (MobileBertConfig, MobileBertModel, MobileBertTokenizer),
    "marian": (MarianConfig, MarianMTModel, MarianTokenizer),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
}


class Seq2SeqModel:
    def __init__(
        self,
        encoder_type=None,
        encoder_name=None,
        decoder_name=None,
        encoder_decoder_type=None,
        encoder_decoder_name=None,
        config=None,
        args=None,
        use_cuda=True,
        prompt_tuning_mode=False,
        ptencoder_name=None,
        static_prompt_tuning_mode=False,
        use_decoder_pt=False,
        num_encoder_prompt_tokens=None,
        num_decoder_prompt_tokens=None,
        pt_dir=None,
        local_rank=None,
        rank=None,
        gpu=None,
        world_size=None,
        dist_url=None,
        dist_backend=None,
        **kwargs,
    ):

        """
        Initializes a Seq2SeqModel.

        Args:
            encoder_type (optional): The type of model to use as the encoder.
            encoder_name (optional): The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
            decoder_name (optional): The exact architecture and trained weights to use. This may be a Hugging Face Transformers compatible pre-trained model, a community model, or the path to a directory containing model files.
                                    Must be the same "size" as the encoder model (base/base, large/large, etc.)
            encoder_decoder_type (optional): The type of encoder-decoder model. (E.g. bart)
            encoder_decoder_name (optional): The path to a directory containing the saved encoder and decoder of a Seq2SeqModel. (E.g. "outputs/") OR a valid BART or MarianMT model.
            config (optional): A configuration file to build an EncoderDecoderModel.
            args (optional): Default args will be used if this parameter is not provided. If provided, it should be a dict containing the args that should be changed in the default args.
            use_cuda (optional): Use GPU if available. Setting to False will force model to use CPU only.
            local_rank (optional): Specific GPU that should be used. Will use the first available GPU by default.
            **kwargs (optional): For providing proxies, force_download, resume_download, cache_dir and other options specific to the 'from_pretrained' implementation where this will be supplied.
        """  # noqa: ignore flake8"

        print("...begin initializing seq2seq model...")
        print("numpy version:", np.__version__)
        print("torch version:", torch.__version__)
        print("transformers version:", transformers.__version__)

        self.prompt_encoder = None   # decide what additional message to give to the PLM
        self.prompt_adapter = None   # adapt the message into PLM's embedding space
        self.prompt_stopper = None   # decide from the prompt encoder's output whether should stop
        self.ptencoder_sep = None    # separator token for prompt encoder

        # DDP configs
        self.local_rank = local_rank
        self.rank = rank
        self.gpu = gpu
        self.world_size = world_size
        self.dist_url = dist_url
        self.dist_backend = dist_backend

        if not config:
            # if not ((encoder_name and decoder_name) or encoder_decoder_name) and not encoder_type:
            if not ((encoder_name and decoder_name) or encoder_decoder_name):
                raise ValueError(
                    "You must specify a Seq2Seq config \t OR \t"
                    "encoder_type, encoder_name, and decoder_name OR \t \t"
                    "encoder_type and encoder_decoder_name"
                )
            elif not (encoder_type or encoder_decoder_type):
                raise ValueError(
                    "You must specify a Seq2Seq config \t OR \t"
                    "encoder_type, encoder_name, and decoder_name \t OR \t"
                    "encoder_type and encoder_decoder_name"
                )

        self.args = self._load_model_args(encoder_decoder_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, Seq2SeqArgs):
            self.args = args
        print(self.args)
        if "sweep_config" in kwargs:
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = {key: value["value"] for key, value in sweep_config.as_dict().items() if key != "_wandb"}
            self.args.update_from_dict(sweep_values)

        # GPU & multi-GPU settings
        self.args.n_gpu = torch.cuda.device_count()
        print("local gpu count:", self.args.n_gpu)
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
        self.distributed = self.world_size > 1

        if self.distributed:
            print("***In distributed mode, world_size:{}***".format(self.world_size))

        if self.distributed:
            if self.local_rank != -1:  # for torch.distributed.launch
                print("provided local_rank is {}. Setting rank and gpu both to be the same.".format(self.local_rank))
                self.rank = self.local_rank
                self.gpu = self.local_rank
            elif 'SLURM_PROCID' in os.environ:  # for slurm scheduler
                self.rank = int(os.environ['SLURM_PROCID'])
                self.gpu = self.rank % self.args.n_gpu
                print("provided local_rank is -1. Setting rank and gpu with SLURM_PROCID. Rank:{}, gpu:{}"
                      .format(self.rank, self.gpu))
            torch.distributed.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                    world_size=self.world_size, rank=self.rank)
            assert self.rank >= 0
        else:
            assert self.rank == -1

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if self.distributed:
            assert use_cuda
        if use_cuda:
            if torch.cuda.is_available():
                if local_rank == -1:
                    self.device = torch.device('cuda')
                else:
                    self.device = torch.device('cuda', local_rank)
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `use_cuda=False`."
                )
        else:
            self.device = "cpu"

        print("setting device complete. device:", self.device)

        self.results = {}

        if not use_cuda:
            self.args.fp16 = False

        # config = EncoderDecoderConfig.from_encoder_decoder_configs(config, config)
        if encoder_decoder_type:
            config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_decoder_type]
        else:
            config_class, model_class, tokenizer_class = MODEL_CLASSES[encoder_type]

        self.model = model_class.from_pretrained(encoder_decoder_name)
        self.encoder_tokenizer = tokenizer_class.from_pretrained(encoder_decoder_name)
        # add the special prompt tokens
        len_before = len(self.encoder_tokenizer)
        self.encoder_tokenizer.add_tokens(['<prompt>'])
        self.pt_token_id = self.encoder_tokenizer.convert_tokens_to_ids('<prompt>')
        len_after = len(self.encoder_tokenizer)
        if len_before == len_after-1:
            print("\t...adding prompt token to model tokenizer...")
        elif len_before == len_after:
            print("\t...prompt token already exists...")
        else:
            assert False

        self.decoder_tokenizer = self.encoder_tokenizer
        self.config = self.model.config

        if self.args.init_model_weights:
            self.model.init_weights()

        if self.args.wandb_project and not wandb_available:
            warnings.warn("wandb_project specified but wandb is not available. Wandb disabled.")
            self.args.wandb_project = None

        # `model_name` could be provided in args
        if self.args.model_name is None:
            if encoder_decoder_name:
                self.args.model_name = encoder_decoder_name

                # # Checking if we are loading from a saved model or using a pre-trained model
                # if not saved_model_args and encoder_decoder_type == "marian":
                # Need to store base pre-trained model name to get the tokenizer when loading a saved model
                self.args.base_marian_model_name = encoder_decoder_name

            elif encoder_name and decoder_name:
                self.args.model_name = encoder_name + "-" + decoder_name
            else:
                self.args.model_name = "encoder-decoder"

            if encoder_decoder_type:
                self.args.model_type = encoder_decoder_type
            elif encoder_type:
                self.args.model_type = encoder_type + "-bert"
            else:
                self.args.model_type = "encoder-decoder"

        ####
        # resize the embeddings since new tokens are perhaps added
        self.model.resize_token_embeddings(len(self.encoder_tokenizer))

        if (not prompt_tuning_mode) and (not static_prompt_tuning_mode):
            assert not use_decoder_pt
        self.prompt_tuning_mode = prompt_tuning_mode
        self.static_prompt_tuning_mode = static_prompt_tuning_mode
        self.use_decoder_pt = use_decoder_pt
        self.num_encoder_prompt_tokens = num_encoder_prompt_tokens
        self.num_decoder_prompt_tokens = num_decoder_prompt_tokens
        if use_decoder_pt:
            # pass some params also to the model
            self.model.n_tokens_decoder, self.model.pt_token_id = num_decoder_prompt_tokens, self.pt_token_id

        if prompt_tuning_mode:
            if use_decoder_pt:
                num_ = self.num_decoder_prompt_tokens
            else:
                num_ = None
            print("In PromptTuning Mode with # of encoder prompt tokens {}, # decoder prompt tokens {}".format(
                self.num_encoder_prompt_tokens, num_))

            print("setting soft prompt embeddings for the model")
            input_embeddings = self.model.get_input_embeddings()
            input_embeddings.requires_grad = False
            # set casual decoder embedding
            if use_decoder_pt:
                self.s_wte_decoder = LEmbedding(
                    input_embeddings, pt_id=self.pt_token_id, n_tokens=self.num_decoder_prompt_tokens
                )
                self.model.set_casual_decoder_embeddings(self.s_wte_decoder)
            else:
                self.model.set_casual_decoder_embeddings(input_embeddings)
            # set the input embedding to be LEmbedding (special forward pass)
            self.s_wte = LEmbedding(
                input_embeddings, pt_id=self.pt_token_id, n_tokens=self.num_encoder_prompt_tokens
            )
            self.model.set_input_embeddings(self.s_wte)

            encoder_special_tokens = ['<en_pt_{}>'.format(i) for i in range(num_encoder_prompt_tokens)]
            decoder_special_tokens = ['<de_pt_{}>'.format(i) for i in range(num_decoder_prompt_tokens)]

            # now load or initialize the prompt encoder/adapter/stopper
            # first decide model/tokenizer class
            if ptencoder_name.startswith('roberta'):
                print("roberta prompter")
                ptencoder_model_class, ptencoder_tokenizer_class = RobertaModel, RobertaTokenizer
            elif ptencoder_name.startswith('google/electra'):
                print("electra prompter")
                ptencoder_model_class, ptencoder_tokenizer_class = ElectraModel, ElectraTokenizer
                self.ptencoder_sep = " [SEP] "
            else:
                print("bert prompter")
                ptencoder_model_class, ptencoder_tokenizer_class = BertModel, BertTokenizer
                self.ptencoder_sep = " [SEP] "

            if os.path.exists(pt_dir + "prompter/"):
                assert os.path.exists(pt_dir+"adapter/")
                assert os.path.exists(pt_dir+"stopper/")
                print("loading pretrained prompt encoder & tokenizer & adapter")
                self.prompt_encoder = ptencoder_model_class.from_pretrained(pt_dir + "prompter/")
                self.ptencoder_tokenizer = ptencoder_tokenizer_class.from_pretrained(pt_dir + "prompter/")
                self.prompt_adapter = torch.load(pt_dir+"adapter/adapter.pth")
                self.prompt_stopper = torch.load(pt_dir+"stopper/stopper.pth")
                print("prompt_encoder & tokenizer embedding shape:",
                      self.prompt_encoder.get_input_embeddings().weight.shape, len(self.ptencoder_tokenizer))
            else:
                print("initializing prompt encoder & tokenizer & adapter & stopper")
                self.prompt_encoder = ptencoder_model_class.from_pretrained(ptencoder_name)
                self.ptencoder_tokenizer = ptencoder_tokenizer_class.from_pretrained(ptencoder_name)
                print("(scratch) prompt_encoder & tokenizer embedding shape:",
                      self.prompt_encoder.get_input_embeddings().weight.shape, len(self.ptencoder_tokenizer))
                # add encoder/decoder special tokens
                self.ptencoder_tokenizer.add_tokens(encoder_special_tokens)
                self.ptencoder_tokenizer.add_tokens(decoder_special_tokens)
                self.prompt_encoder.resize_token_embeddings(len(self.ptencoder_tokenizer))
                shape_ptencoder_embedding = self.prompt_encoder.get_input_embeddings().weight.shape
                print("prompt_encoder & tokenizer embedding shape:",
                      shape_ptencoder_embedding, len(self.ptencoder_tokenizer))

                d_ptencoder = shape_ptencoder_embedding[1]
                self.prompt_adapter = FeedForwardNet(d_ptencoder, 1024)
                self.prompt_stopper = FeedForwardNet(d_ptencoder, 2, hid_layer=[500, 100])

            self.en_special_token_ids = self.ptencoder_tokenizer.convert_tokens_to_ids(encoder_special_tokens)
            self.de_special_token_ids = self.ptencoder_tokenizer.convert_tokens_to_ids(decoder_special_tokens)

            # the chunk of special token ids to be added immediately after the BOS token for ptencoder inputs
            # a 1-d tensor on cpu.
            if use_decoder_pt:
                self.ptencoder_insert_ids = torch.tensor(
                    self.en_special_token_ids[:num_encoder_prompt_tokens] + self.de_special_token_ids[:num_decoder_prompt_tokens])
            else:
                self.ptencoder_insert_ids = torch.tensor(self.en_special_token_ids[:num_encoder_prompt_tokens])
        elif static_prompt_tuning_mode:
            if use_decoder_pt:
                num_ = self.num_decoder_prompt_tokens
            else:
                num_ = None
            print("In Static PromptTuning Mode with # of encoder prompt tokens {}, # decoder prompt tokens {}".format(
                self.num_encoder_prompt_tokens, num_))

            print("setting soft prompt embeddings for the model")
            input_embeddings = self.model.get_input_embeddings()
            input_embeddings.requires_grad = False
            # set casual decoder embedding
            if use_decoder_pt:
                if os.path.exists(pt_dir + "static_prompt_vectors/"):
                    print("loading pretrained decoder prompt vectors")
                    self.s_wte_decoder = SoftEmbedding(input_embeddings, pt_id=self.pt_token_id,
                        n_tokens=self.num_decoder_prompt_tokens, load_from_path=pt_dir+"static_prompt_vectors/de.pt")
                else:
                    print("init decoder prompt vectors")
                    self.s_wte_decoder = SoftEmbedding(input_embeddings, pt_id=self.pt_token_id,
                                                       n_tokens=self.num_decoder_prompt_tokens)
                self.model.set_casual_decoder_embeddings(self.s_wte_decoder)
            else:
                self.model.set_casual_decoder_embeddings(input_embeddings)
            # set the input embedding
            if os.path.exists(pt_dir + "static_prompt_vectors/"):
                print("loading pretrained encoder prompt vectors")
                self.s_wte = SoftEmbedding(input_embeddings, pt_id=self.pt_token_id,
                        n_tokens=self.num_encoder_prompt_tokens, load_from_path=pt_dir+"static_prompt_vectors/en.pt")
            else:
                print("init encoder prompt vectors")
                self.s_wte = SoftEmbedding(input_embeddings, pt_id=self.pt_token_id,
                                       n_tokens=self.num_encoder_prompt_tokens)
            self.model.set_input_embeddings(self.s_wte)
        else:
            print("In ModelTuning Mode.")

        print("initializing model complete; number of gpus:", self.args.n_gpu)
        print("# model params:", sum(p.numel() for p in self.model.parameters()))
        if self.prompt_tuning_mode:
            print("# prompter params:", sum(p.numel() for p in self.prompt_encoder.parameters()))
            if not(self.prompt_adapter is None):
                print("# adapter params:", sum(p.numel() for p in self.prompt_adapter.parameters()))
            if not(self.prompt_stopper is None):
                print("# stopper params:", sum(p.numel() for p in self.prompt_stopper.parameters()))

    def train_model(
        self, train_data, output_dir=None, show_running_loss=True, args=None, eval_data=None, test_data=None,
            verbose=True, train_stopper=False, **kwargs,
    ):
        if not self.prompt_tuning_mode:
            assert not train_stopper
        """
        Trains the model using 'train_data'

        Args:
            train_data: Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
                        - `input_text`: The input text sequence.
                        - `target_text`: The target text sequence
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_data (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            test_data (optional): Test data which prediction will be performed when predict_during_training is enabled. Is required if prediction_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down training significantly as the predicted sequences need to be generated.

        Returns:
            None
        """  # noqa: ignore flake8"
        tick = time.time()

        if args:
            self.args.update_from_dict(args)

        # if self.args.silent:
        #     show_running_loss = False

        if self.args.evaluate_during_training and eval_data is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_data is not specified."
                " Pass eval_data to model.train_model() if using evaluate_during_training."
            )

        if self.args.predict_during_training and test_data is None:
            raise ValueError(
                "predict_during_training is enabled but test_data is not specified."
                " Pass test_data to model.train_model() if using predict_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set args.overwrite_output_dir = True to overcome.".format(output_dir)
            )

        """
        if self.args.n_gpu > 1:
            torch.distributed.init_process_group('nccl')  # DDP
        """
        self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(train_data, verbose=verbose)
        tock = time.time()
        print("caching examples using {} seconds".format(tock-tick))

        os.makedirs(output_dir, exist_ok=True)
        if self.prompt_tuning_mode:
            os.makedirs(os.path.join(output_dir, "prompter/"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "adapter/"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "stopper/"), exist_ok=True)

        global_step, tr_loss = self.train(
            train_dataset,
            output_dir,
            show_running_loss=show_running_loss,
            eval_data=eval_data,
            test_data=test_data,
            verbose=verbose,
            train_stopper=train_stopper,
            **kwargs,
        )

        if self.prompt_tuning_mode:
            self.save_prompter(self.args.output_dir)
            self.save_adapter(self.args.output_dir)
            self.save_stopper(self.args.output_dir)
        elif self.static_prompt_tuning_mode:
            self.save_static_prompt_vectors(self.args.output_dir)
        else:
            self.save_model(self.args.output_dir, model=self.model)

        # model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        # model_to_save.save_pretrained(output_dir)
        # self.encoder_tokenizer.save_pretrained(output_dir)
        # self.decoder_tokenizer.save_pretrained(output_dir)
        # torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        if verbose:
            logger.info(" Training of {} model complete. Saved to {}.".format(self.args.model_name, output_dir))

    def train(
        self, train_dataset, output_dir, show_running_loss=True, eval_data=None, test_data=None,
            verbose=True, train_stopper=False, **kwargs,
    ):
        """
        Trains the model on train_dataset.
        Utility function to be used by the train_model() method. Not intended to be used directly.
        """
        model = self.model
        args = self.args

        tb_writer = SummaryWriter(logdir=args.tensorboard_dir)
        train_sampler = RandomSampler(train_dataset)
        if self.distributed:
            print("invoking distributed sampler for rank", self.rank)
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        no_decay = ["bias", "LayerNorm.weight"]

        if self.prompt_tuning_mode:
            # now freeze all model params except the prompt encoder's
            parameters = list(model.parameters())
            for x in parameters:
                x.requires_grad = False

            # add prompt encoder params
            optimizer_grouped_parameters = []
            custom_parameter_names = set()
            for group in self.args.custom_parameter_groups:
                params = group.pop("params")
                custom_parameter_names.update(params)
                param_group = {**group}
                param_group["params"] = [p for n, p in self.prompt_encoder.named_parameters() if n in params]
                optimizer_grouped_parameters.append(param_group)

            for group in self.args.custom_layer_parameters:
                layer_number = group.pop("layer")
                layer = f"layer.{layer_number}."
                group_d = {**group}
                group_nd = {**group}
                group_nd["weight_decay"] = 0.0
                params_d = []
                params_nd = []
                for n, p in self.prompt_encoder.named_parameters():
                    if n not in custom_parameter_names and layer in n:
                        if any(nd in n for nd in no_decay):
                            params_nd.append(p)
                        else:
                            params_d.append(p)
                        custom_parameter_names.add(n)
                group_d["params"] = params_d
                group_nd["params"] = params_nd

                optimizer_grouped_parameters.append(group_d)
                optimizer_grouped_parameters.append(group_nd)

            if not self.args.train_custom_parameters_only:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [
                                p
                                for n, p in self.prompt_encoder.named_parameters()
                                if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                            ],
                            "weight_decay": args.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in self.prompt_encoder.named_parameters()
                                if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                            ],
                            "weight_decay": 0.0,
                        },
                    ]
                )
            # add adapter and stopper params
            optimizer_grouped_parameters.append({'params': self.prompt_adapter.parameters()})
            if train_stopper:
                optimizer_grouped_parameters.append({'params': self.prompt_stopper.parameters()})

        elif self.static_prompt_tuning_mode:
            parameters = list(model.parameters())
            if self.use_decoder_pt:
                for x in parameters[1:-1]:
                    x.requires_grad = False
            else:
                for x in parameters[1:]:
                    x.requires_grad = False
            optimizer_grouped_parameters = []
            for parameter in list(model.parameters()):
                if parameter.requires_grad:
                    optimizer_grouped_parameters.append(parameter)
            print("shapes of parameters to be optimized:")
            for param in optimizer_grouped_parameters:
                print(param.shape, end=" ")
            print()
            print("total number of parameters to be optimized", len(optimizer_grouped_parameters))
        else:
            optimizer_grouped_parameters = []
            custom_parameter_names = set()
            for group in self.args.custom_parameter_groups:
                params = group.pop("params")
                custom_parameter_names.update(params)
                param_group = {**group}
                param_group["params"] = [p for n, p in model.named_parameters() if n in params]
                optimizer_grouped_parameters.append(param_group)

            for group in self.args.custom_layer_parameters:
                layer_number = group.pop("layer")
                layer = f"layer.{layer_number}."
                group_d = {**group}
                group_nd = {**group}
                group_nd["weight_decay"] = 0.0
                params_d = []
                params_nd = []
                for n, p in model.named_parameters():
                    if n not in custom_parameter_names and layer in n:
                        if any(nd in n for nd in no_decay):
                            params_nd.append(p)
                        else:
                            params_d.append(p)
                        custom_parameter_names.add(n)
                group_d["params"] = params_d
                group_nd["params"] = params_nd

                optimizer_grouped_parameters.append(group_d)
                optimizer_grouped_parameters.append(group_nd)

            if not self.args.train_custom_parameters_only:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [
                                p
                                for n, p in model.named_parameters()
                                if n not in custom_parameter_names and not any(nd in n for nd in no_decay)
                            ],
                            "weight_decay": args.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in model.named_parameters()
                                if n not in custom_parameter_names and any(nd in n for nd in no_decay)
                            ],
                            "weight_decay": 0.0,
                        },
                    ]
                )

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = warmup_steps if args.warmup_steps == 0 else args.warmup_steps

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        if (
            args.model_name
            and os.path.isfile(os.path.join(args.model_name, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name, "scheduler.pt"))
        ):
            print("Load in optimizer and scheduler states")
            optimizer.load_state_dict(torch.load(os.path.join(args.model_name, "optimizer.pt")))
            scheduler.load_state_dict(torch.load(os.path.join(args.model_name, "scheduler.pt")))

        if self.distributed:
            # DDP
            if self.local_rank == -1:
                temp = 0
            else:
                temp = self.local_rank
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[temp], output_device=temp)

        # in the distributed case, disable prints for non-master nodes
        if self.distributed:
            if self.rank != 0:
                print("I'm rank {}. I'm muted from now on.".format(self.rank))
                def print_pass(*args_):
                    pass
                builtins.print = print_pass
            else:
                print("I'm rank {}. I'll continue to print.".format(self.rank))

        logger.info(" Training started")
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        if self.prompt_tuning_mode:
            self.prompt_encoder.zero_grad()
            self.prompt_adapter.zero_grad()
            if train_stopper:
                self.prompt_stopper.zero_grad()
        elif self.static_prompt_tuning_mode:
            optimizer.zero_grad()
        else:
            model.zero_grad()
        train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0)
        epoch_number = 0
        best_eval_metric = None
        early_stopping_counter = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if args.model_name and os.path.exists(args.model_name):
            try:
                print("try set global_step to gobal_step of last saved checkpoint from model path")
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )

                logger.info("   Continuing training from checkpoint, will skip to saved global_step")
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info("   Will skip the first %d steps in the current epoch", steps_trained_in_current_epoch)
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(**kwargs)

        if args.wandb_project:
            wandb.init(project=args.wandb_project, config={**asdict(args)}, **args.wandb_kwargs)
            wandb.watch(self.model)

        if args.fp16:
            from torch.cuda import amp
            scaler = amp.GradScaler()

        if self.prompt_tuning_mode:
            model.eval()     # we don't want dropouts in pt mode
            self.prompt_encoder.train()
            self.prompt_adapter.train()
            if train_stopper:
                self.prompt_stopper.train()
        else:
            model.train()
        for current_epoch in train_iterator:
            if self.distributed:
                train_dataloader.sampler.set_epoch(current_epoch)
            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(f"Epoch {epoch_number + 1} of {args.num_train_epochs}")
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )
            total_loss = 0
            if train_stopper:
                stopper_total_loss = 0
            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                # batch = tuple(t.to(device) for t in batch)
                inputs, ptencoder_inputs, stopper_y = self._get_inputs_dict(batch)
                if args.fp16:
                    with amp.autocast():
                        if self.prompt_tuning_mode:
                            ptencoder_last_states = self.prompt_encoder(**ptencoder_inputs).last_hidden_state
                            if train_stopper:
                                stopper_logits = self.prompt_stopper(ptencoder_last_states[:, 0, :])
                                stopper_loss = nn.CrossEntropyLoss()(stopper_logits, stopper_y)
                            temp = ptencoder_last_states[:, 1:1+self.num_encoder_prompt_tokens, :]
                            self.s_wte.learned_embedding = self.prompt_adapter(temp)
                            if self.use_decoder_pt:
                                temp = ptencoder_last_states[:, 1+self.num_encoder_prompt_tokens: 1+self.num_encoder_prompt_tokens+self.num_decoder_prompt_tokens, :]
                                self.s_wte_decoder.learned_embedding = self.prompt_adapter(temp)
                            loss = model(**inputs)[0]
                            self.s_wte.learned_embedding = None
                            if self.use_decoder_pt:
                                self.s_wte_decoder.learned_embedding = None
                        else:
                            loss = model(**inputs)[0]
                else:
                    if self.prompt_tuning_mode:
                        ptencoder_last_states = self.prompt_encoder(**ptencoder_inputs).last_hidden_state
                        if train_stopper:
                            stopper_logits = self.prompt_stopper(ptencoder_last_states[:, 0, :])
                            stopper_loss = nn.CrossEntropyLoss()(stopper_logits, stopper_y)
                        temp = ptencoder_last_states[:, 1:1 + self.num_encoder_prompt_tokens, :]
                        self.s_wte.learned_embedding = self.prompt_adapter(temp)
                        if self.use_decoder_pt:
                            temp = ptencoder_last_states[:,1 + self.num_encoder_prompt_tokens: 1 + self.num_encoder_prompt_tokens + self.num_decoder_prompt_tokens,:]
                            self.s_wte_decoder.learned_embedding = self.prompt_adapter(temp)
                        loss = model(**inputs)[0]
                        self.s_wte.learned_embedding = None
                        if self.use_decoder_pt:
                            self.s_wte_decoder.learned_embedding = None
                    else:
                        loss = model(**inputs)[0]

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training

                current_loss = loss.item()
                total_loss += current_loss
                if train_stopper:
                    current_stopper_loss = stopper_loss.item()
                    stopper_total_loss += current_stopper_loss

                if show_running_loss:
                    if self.prompt_tuning_mode and train_stopper:
                        batch_iterator.set_description(
                            f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}; Stopper Loss: {current_stopper_loss:9.4f}"
                        )
                    else:
                        batch_iterator.set_description(
                            f"Epochs {epoch_number}/{args.num_train_epochs}. Running Loss: {current_loss:9.4f}"
                        )

                if self.prompt_tuning_mode and train_stopper:
                    loss = loss + 0.1 * stopper_loss       # TODO: hyper-params

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # gradient step
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    if self.prompt_tuning_mode:
                        torch.nn.utils.clip_grad_norm_(self.prompt_encoder.parameters(), args.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(self.prompt_adapter.parameters(), args.max_grad_norm)
                        if train_stopper:
                            torch.nn.utils.clip_grad_norm_(self.prompt_stopper.parameters(), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    if args.fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()     # Update learning rate schedule
                    if self.prompt_tuning_mode:
                        self.prompt_encoder.zero_grad()
                        self.prompt_adapter.zero_grad()
                        if train_stopper:
                            self.prompt_stopper.zero_grad()
                    elif self.static_prompt_tuning_mode:
                        optimizer.zero_grad()
                    else:
                        model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        # Log metrics
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss
                        if args.wandb_project:
                            wandb.log(
                                {
                                    "Training loss": current_loss,
                                    "lr": scheduler.get_lr()[0],
                                    "global_step": global_step,
                                }
                            )

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        # Save model checkpoint
                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        if self.prompt_tuning_mode:
                            self.save_prompter(output_dir_current)
                            self.save_adapter(output_dir_current)
                            self.save_stopper(output_dir_current)
                        elif self.static_prompt_tuning_mode:
                            self.save_static_prompt_vectors(output_dir_current)
                        else:
                            self.save_model(output_dir_current, optimizer, scheduler, model=model)

                        if args.predict_during_training:
                            self.predict(test_data, output_dir_current, suffix=str(global_step))


                    if args.evaluate_during_training and (
                        args.evaluate_during_training_steps > 0
                        and global_step % args.evaluate_during_training_steps == 0
                    ):
                        # Only evaluate when single GPU otherwise metrics may not average well
                        results = self.eval_model(
                            eval_data,
                            verbose=verbose and args.evaluate_during_training_verbose,
                            silent=args.evaluate_during_training_silent,
                            **kwargs,
                        )
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                        output_dir_current = os.path.join(output_dir, "checkpoint-{}".format(global_step))

                        if args.save_eval_checkpoints:

                            if self.prompt_tuning_mode:
                                self.save_prompter(output_dir_current)
                                self.save_adapter(output_dir_current)
                                self.save_stopper(output_dir_current)
                            elif self.static_prompt_tuning_mode:
                                self.save_static_prompt_vectors(output_dir_current)
                            else:
                                self.save_model(output_dir_current, optimizer, scheduler, model=model, results=results)

                        training_progress_scores["global_step"].append(global_step)
                        training_progress_scores["train_loss"].append(current_loss)
                        for key in results:
                            training_progress_scores[key].append(results[key])
                        report = pd.DataFrame(training_progress_scores)
                        report.to_csv(
                            os.path.join(args.output_dir, "training_progress_scores.csv"), index=False,
                        )

                        if args.wandb_project:
                            wandb.log(self._get_last_metrics(training_progress_scores))

                        if not best_eval_metric:
                            best_eval_metric = results[args.early_stopping_metric]
                            if args.save_best_model:

                                if self.prompt_tuning_mode:
                                    self.save_prompter(args.output_dir)
                                    self.save_adapter(args.output_dir)
                                    self.save_stopper(args.output_dir)
                                elif self.static_prompt_tuning_mode:
                                    self.save_static_prompt_vectors(args.output_dir)
                                else:
                                    self.save_model(args.output_dir, optimizer, scheduler, model=model, results=results)

                        if best_eval_metric and args.early_stopping_metric_minimize:
                            if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                if args.save_best_model:

                                    if self.prompt_tuning_mode:
                                        self.save_prompter(args.output_dir)
                                        self.save_adapter(args.output_dir)
                                        self.save_stopper(args.output_dir)
                                    elif self.static_prompt_tuning_mode:
                                        self.save_static_prompt_vectors(args.output_dir)
                                    else:
                                        self.save_model(args.output_dir, optimizer, scheduler, model=model, results=results)

                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return global_step, tr_loss / global_step
                        else:
                            if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                                best_eval_metric = results[args.early_stopping_metric]
                                if args.save_best_model:

                                    if self.prompt_tuning_mode:
                                        self.save_prompter(args.output_dir)
                                        self.save_adapter(args.output_dir)
                                        self.save_stopper(args.output_dir)
                                    elif self.static_prompt_tuning_mode:
                                        self.save_static_prompt_vectors(args.output_dir)
                                    else:
                                        self.save_model(args.output_dir, optimizer, scheduler, model=model,  results=results)

                                early_stopping_counter = 0
                            else:
                                if args.use_early_stopping:
                                    if early_stopping_counter < args.early_stopping_patience:
                                        early_stopping_counter += 1
                                        if verbose:
                                            logger.info(f" No improvement in {args.early_stopping_metric}")
                                            logger.info(f" Current step: {early_stopping_counter}")
                                            logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                                    else:
                                        if verbose:
                                            logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                            logger.info(" Training terminated.")
                                            train_iterator.close()
                                        return global_step, tr_loss / global_step

                # return   # TODO: remove
            if self.prompt_tuning_mode and train_stopper:
                print("epoch {}, total loss:{}, total stopper loss:{}".format(current_epoch, total_loss, stopper_total_loss))
            else:
                print("epoch {}, total loss:{}".format(current_epoch, total_loss))
            if self.prompt_tuning_mode:
                for x in model.parameters():
                    assert not x.requires_grad

            epoch_number += 1
            output_dir_current = os.path.join(output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number))

            if args.save_model_every_epoch or args.evaluate_during_training:
                os.makedirs(output_dir_current, exist_ok=True)
                if self.prompt_tuning_mode:
                    os.makedirs(os.path.join(output_dir_current, "prompter/"), exist_ok=True)
                    os.makedirs(os.path.join(output_dir_current, "adapter/"), exist_ok=True)
                    os.makedirs(os.path.join(output_dir_current, "stopper/"), exist_ok=True)
                if self.static_prompt_tuning_mode:
                    os.makedirs(os.path.join(output_dir_current, "static_prompt_vectors/"), exist_ok=True)

            if args.save_model_every_epoch:

                if self.prompt_tuning_mode:
                    self.save_prompter(output_dir_current)
                    self.save_adapter(output_dir_current)
                    self.save_stopper(output_dir_current)
                elif self.static_prompt_tuning_mode:
                    self.save_static_prompt_vectors(output_dir_current)
                else:
                    self.save_model(output_dir_current, optimizer, scheduler, model=model)

                if args.predict_during_training:
                    self.predict(test_data, output_dir_current, suffix=str(global_step))

            if args.evaluate_during_training:
                results = self.eval_model(
                    eval_data,
                    verbose=verbose and args.evaluate_during_training_verbose,
                    silent=args.evaluate_during_training_silent,
                    **kwargs,
                )

                if args.save_eval_checkpoints:
                    if self.prompt_tuning_mode:
                        self.save_prompter(output_dir_current)
                        self.save_adapter(output_dir_current)
                        self.save_stopper(output_dir_current)
                    elif self.static_prompt_tuning_mode:
                        self.save_static_prompt_vectors(output_dir_current)
                    else:
                        self.save_model(output_dir_current, optimizer, scheduler, results=results)

                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["train_loss"].append(current_loss)
                for key in results:
                    training_progress_scores[key].append(results[key])
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(os.path.join(args.output_dir, "training_progress_scores.csv"), index=False)

                if args.wandb_project:
                    wandb.log(self._get_last_metrics(training_progress_scores))

                if not best_eval_metric:
                    best_eval_metric = results[args.early_stopping_metric]
                    if args.save_best_model:

                        if self.prompt_tuning_mode:
                            self.save_prompter(args.output_dir)
                            self.save_adapter(args.output_dir)
                            self.save_stopper(args.output_dir)
                        elif self.static_prompt_tuning_mode:
                            self.save_static_prompt_vectors(args.output_dir)
                        else:
                            self.save_model(args.output_dir, optimizer, scheduler, model=model, results=results)

                if best_eval_metric and args.early_stopping_metric_minimize:
                    if results[args.early_stopping_metric] - best_eval_metric < args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        if args.save_best_model:

                            if self.prompt_tuning_mode:
                                self.save_prompter(args.output_dir)
                                self.save_adapter(args.output_dir)
                                self.save_stopper(args.output_dir)
                            elif self.static_prompt_tuning_mode:
                                self.save_static_prompt_vectors(args.output_dir)
                            else:
                                self.save_model(args.output_dir, optimizer, scheduler, model=model, results=results)

                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return global_step, tr_loss / global_step
                else:
                    if results[args.early_stopping_metric] - best_eval_metric > args.early_stopping_delta:
                        best_eval_metric = results[args.early_stopping_metric]
                        if args.save_best_model:

                            if self.prompt_tuning_mode:
                                self.save_prompter(args.output_dir)
                                self.save_adapter(args.output_dir)
                                self.save_stopper(args.output_dir)
                            elif self.static_prompt_tuning_mode:
                                self.save_static_prompt_vectors(args.output_dir)
                            else:
                                self.save_model(args.output_dir, optimizer, scheduler, model=model, results=results)

                        early_stopping_counter = 0
                    else:
                        if args.use_early_stopping and args.early_stopping_consider_epochs:
                            if early_stopping_counter < args.early_stopping_patience:
                                early_stopping_counter += 1
                                if verbose:
                                    logger.info(f" No improvement in {args.early_stopping_metric}")
                                    logger.info(f" Current step: {early_stopping_counter}")
                                    logger.info(f" Early stopping patience: {args.early_stopping_patience}")
                            else:
                                if verbose:
                                    logger.info(f" Patience of {args.early_stopping_patience} steps reached")
                                    logger.info(" Training terminated.")
                                    train_iterator.close()
                                return global_step, tr_loss / global_step

        return global_step, tr_loss / global_step

    def predict(self, pred_data, output_dir=None, suffix=None, verbose=True, silent=False,
                iter_step=1, auto_stop=False, show_pt_nns=False):
        """
        Performs predictions on a list of text.
        Args:
            pred_data: Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
                        - `input_text`: The input text sequence.
                        - `target_text`: The target text sequence.            
            output_dir: The directory where predictition results files will be saved. If not given, self.args.output_dir will be used.
            suffix: The supplementary suffix of prediction results name.
            iter_step: # of steps for iterative generation. Default is 1 (normal generation).
            auto_stop (only available in prompt tuning mode): whether use the stopper to decide whether to stop. 
                If yes, iter_step becomes the maximum step. 
        Returns:
            preds: A python list of the generated sequences.
        """  # noqa: ignore flake8"

        self.model.eval()
        if self.prompt_tuning_mode:
            self.prompt_encoder.eval()
            self.prompt_adapter.eval()
            self.prompt_stopper.eval()

        to_predict = pred_data["input_text"].tolist()
        target_predict = pred_data["target_text"].tolist()

        assert len(to_predict) == len(target_predict)

        self._move_model_to_device()

        if not output_dir:
            output_dir = self.args.output_dir

        all_outputs = []
        # Batching
        for batch in tqdm([to_predict[i: i + self.args.eval_batch_size] for i in range(0, len(to_predict), self.args.eval_batch_size)],
                          desc='Predicting', disable=self.args.silent, mininterval=0,):
            for i in range(len(batch)):
                batch[i] = batch[i].strip("\n")
            all_outputs_batch = ["" for _ in range(len(batch))]

            curr_step = 0
            while True:
                curr_step += 1
                if curr_step > iter_step:
                    break
                # ========= batch -> input_ids ===========

                input_ids = self.encoder_tokenizer.batch_encode_plus(
                    batch,
                    max_length=self.args.max_seq_length,
                    padding=True,
                    return_tensors="pt",
                    truncation=True,
                )["input_ids"]
                ####
                if self.prompt_tuning_mode or self.static_prompt_tuning_mode:
                    # add the prompt tokens to generation input_ids
                    prompt_chunk = torch.full((input_ids.shape[0], self.num_encoder_prompt_tokens), self.pt_token_id, dtype=torch.int64)
                    input_ids = torch.cat([input_ids[:, :1], prompt_chunk, input_ids[:, 1:]], 1)
                if self.prompt_tuning_mode:
                    if not (self.ptencoder_sep is None):
                        batch_ptencoder = []
                        for line in batch:
                            batch_ptencoder.append(line.replace("</s></s>", self.ptencoder_sep))
                    else:
                        batch_ptencoder = batch

                    ptencoder_inputs = self.ptencoder_tokenizer.batch_encode_plus(
                        batch_ptencoder, max_length=self.args.max_seq_length, return_tensors="pt", padding=True, truncation=True)
                    ptencoder_input_ids = ptencoder_inputs["input_ids"]
                    ptencoder_chunk = self.ptencoder_insert_ids.repeat(ptencoder_input_ids.shape[0], 1)
                    ptencoder_input_ids = torch.cat([ptencoder_input_ids[:, :1], ptencoder_chunk, ptencoder_input_ids[:, 1:]], 1)
                    ptencoder_attention_mask = (ptencoder_input_ids != self.ptencoder_tokenizer.pad_token_id).to(torch.float32)
                    ptencoder_inputs["input_ids"] = ptencoder_input_ids
                    ptencoder_inputs["attention_mask"] = ptencoder_attention_mask
                    if not (self.ptencoder_sep is None):
                        ptencoder_inputs["token_type_ids"] = torch.zeros_like(ptencoder_input_ids)

                input_ids = input_ids.to(self.device)
                if self.prompt_tuning_mode:
                    for key in ptencoder_inputs.keys():
                        ptencoder_inputs[key] = ptencoder_inputs[key].to(self.device)

                # ========= input_ids -> outputs_batch (list of output ids in the batch) ===============
                if self.prompt_tuning_mode:
                    # prompt encoder forward pass
                    ptencoder_last_states = self.prompt_encoder(**ptencoder_inputs).last_hidden_state
                    if auto_stop:
                        # see if stopper says stop (it doesn't actually stop, but rather just put a special marker)
                        stopper_logits = self.prompt_stopper(ptencoder_last_states[:, 0, :])
                        if_stop = torch.argmax(stopper_logits, dim=1)
                        for temp_i in range(len(batch)):
                            if if_stop[temp_i] == 1:
                                all_outputs_batch[temp_i] = all_outputs_batch[temp_i] + "<stopper_stop>"
                    temp = ptencoder_last_states[:, 1:1 + self.num_encoder_prompt_tokens, :]
                    self.s_wte.learned_embedding = self.prompt_adapter(temp)
                    if self.use_decoder_pt:
                        temp = ptencoder_last_states[:,1 + self.num_encoder_prompt_tokens: 1 + self.num_encoder_prompt_tokens + self.num_decoder_prompt_tokens,:]
                        self.s_wte_decoder.learned_embedding = self.prompt_adapter(temp)

                if (not self.prompt_tuning_mode) and (not self.static_prompt_tuning_mode):
                    assert not self.use_decoder_pt

                outputs = self.model.generate(
                    input_ids=input_ids,
                    num_beams=self.args.num_beams,
                    max_length=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    early_stopping=self.args.early_stopping,
                    repetition_penalty=self.args.repetition_penalty,
                    do_sample=self.args.do_sample,
                    top_k=self.args.top_k,
                    top_p=self.args.top_p,
                    num_return_sequences=self.args.num_return_sequences,
                    use_casual_decoder_embeddings=self.prompt_tuning_mode or self.static_prompt_tuning_mode,
                    insert_decoder_prompt=self.use_decoder_pt,
                )
                if self.prompt_tuning_mode:
                    self.s_wte.learned_embedding = None
                    if self.use_decoder_pt:
                        self.s_wte_decoder.learned_embedding = None

                # print(outputs.shape)
                # print("---")
                # print(self.encoder_tokenizer.decode(outputs[0]))
                # print(self.encoder_tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True))
                # print(self.args.num_beams, self.args.max_length, self.args.length_penalty, self.args.early_stopping, self.args.repetition_penalty,
                #      self.args.do_sample, self.args.top_k, self.args.top_p, self.args.num_return_sequences)
                # 1 64 2.0 True 1.0 False None None 1

                outputs_batch = outputs.cpu().numpy()
                # return     # TODO: remove

                # ========= outputs_batch -> outputs (list of output strings in the batch) ===============
                if self.args.use_multiprocessed_decoding:
                    self.model.to("cpu")
                    with Pool(self.args.process_count) as p:
                        outputs = list(
                            tqdm(
                                p.imap(self._decode, outputs_batch, chunksize=self.args.multiprocessing_chunksize),
                                total=len(outputs_batch),
                                desc="Decoding outputs",
                                disable=self.args.silent,
                            )
                        )
                    self._move_model_to_device()
                else:
                    outputs = [
                        self.decoder_tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        for output_id in outputs_batch
                    ]

                # === append the generated output to the final result and also the input for next step generation ===
                for temp_i in range(len(all_outputs_batch)):
                    all_outputs_batch[temp_i] = all_outputs_batch[temp_i] + outputs[temp_i] + "<special_sep>"
                for temp_i in range(len(batch)):
                    batch[temp_i] = batch[temp_i] + '</s></s>' + outputs[temp_i]

            all_outputs.extend(all_outputs_batch)
        outputs = all_outputs

        # write prediction results to file
        os.makedirs(output_dir, exist_ok=True)
        output_predication_file = os.path.join(output_dir, "predictions_{}.txt".format(suffix))
        correct_num = 0
        accuracy_list = []
        with open(output_predication_file, "w", encoding="utf8", errors="ignore") as writer:
            writer.write("to_predict\n\toutput\n\ttarget\n")
            for i in range(len(outputs)):
                outputs[i] = outputs[i].strip().replace("\n", " ")
                writer.write(to_predict[i]+"\t"+outputs[i]+"\n\t"+target_predict[i])

        if self.args.num_return_sequences > 1:
            return [
                outputs[i: i + self.args.num_return_sequences]
                for i in range(0, len(outputs), self.args.num_return_sequences)
            ]
        else:
            return outputs

    def _decode(self, output_id):
        return self.decoder_tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def load_and_cache_examples(self, data, evaluate=False, no_cache=False, verbose=True, silent=False):
        """
        Creates a T5Dataset from data.
        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        encoder_tokenizer = self.encoder_tokenizer
        decoder_tokenizer = self.decoder_tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"

        if args.dataset_class:
            CustomDataset = args.dataset_class
            return CustomDataset(encoder_tokenizer, decoder_tokenizer, args, data, mode)
        else:
            if self.prompt_tuning_mode:
                return SimpleSummarizationDataset(encoder_tokenizer, self.args, data, mode,
                    pt_mode=self.prompt_tuning_mode,
                    static_pt_mode=self.static_prompt_tuning_mode,
                    n_en_pt_tokens=self.num_encoder_prompt_tokens,
                    n_de_pt_tokens=self.num_decoder_prompt_tokens,
                    pt_token_id=self.pt_token_id, use_decoder_pt=self.use_decoder_pt,
                    ptencoder_insert_ids=self.ptencoder_insert_ids,
                    ptencoder_tokenizer=self.ptencoder_tokenizer,
                    ptencoder_sep=self.ptencoder_sep,)
            elif self.static_prompt_tuning_mode:
                return SimpleSummarizationDataset(encoder_tokenizer, self.args, data, mode,
                    pt_mode=self.prompt_tuning_mode,
                    static_pt_mode=self.static_prompt_tuning_mode,
                    n_en_pt_tokens=self.num_encoder_prompt_tokens,
                    n_de_pt_tokens=self.num_decoder_prompt_tokens,
                    pt_token_id=self.pt_token_id, use_decoder_pt=self.use_decoder_pt)
            else:
                return SimpleSummarizationDataset(encoder_tokenizer, self.args, data, mode)

    def _create_training_progress_scores(self, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "eval_loss": [],
            "train_loss": [],
            **extra_metrics,
        }

        return training_progress_scores

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def save_prompter(self, output_dir):
        path_ = os.path.join(output_dir, "prompter/")
        os.makedirs(path_, exist_ok=True)
        self.prompt_encoder.save_pretrained(path_)
        self.ptencoder_tokenizer.save_pretrained(path_)

    def save_adapter(self, output_dir):
        os.makedirs(os.path.join(output_dir, "adapter/"), exist_ok=True)
        path_ = os.path.join(output_dir, "adapter/adapter.pth")
        torch.save(self.prompt_adapter, path_)

    def save_stopper(self, output_dir):
        os.makedirs(os.path.join(output_dir, "stopper/"), exist_ok=True)
        path_ = os.path.join(output_dir, "stopper/stopper.pth")
        torch.save(self.prompt_stopper, path_)

    def save_static_prompt_vectors(self, output_dir):
        assert self.static_prompt_tuning_mode
        os.makedirs(os.path.join(output_dir, "static_prompt_vectors/"), exist_ok=True)
        torch.save(self.s_wte.learned_embedding, os.path.join(output_dir, "static_prompt_vectors/en.pt"))
        if self.use_decoder_pt:
            torch.save(self.s_wte_decoder.learned_embedding, os.path.join(output_dir, "static_prompt_vectors/de.pt"))

    def save_model(self, output_dir=None, optimizer=None, scheduler=None, model=None, results=None):

        if self.distributed:
            if self.rank != 0:
                return

        assert not self.prompt_tuning_mode
        assert not self.static_prompt_tuning_mode
        if not output_dir:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving model into {output_dir}")

        if model and not self.args.no_save:
            # Take care of distributed/parallel training
            if hasattr(model, "module"):
                model_to_save = model.module
            else:
                model_to_save = model
            self.save_model_args(output_dir)

            os.makedirs(os.path.join(output_dir), exist_ok=True)
            if self.prompt_tuning_mode:
                os.makedirs(os.path.join(output_dir, "prompter/"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "adapter/"), exist_ok=True)
                os.makedirs(os.path.join(output_dir, "stopper/"), exist_ok=True)
            model_to_save.save_pretrained(output_dir)
            self.config.save_pretrained(output_dir)
            if self.args.model_type == "bart":
                self.encoder_tokenizer.save_pretrained(output_dir)


            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def _move_model_to_device(self):
        self.model.to(self.device)
        if not(self.prompt_encoder is None):
            self.prompt_encoder.to(self.device)
        if not(self.prompt_adapter is None):
            self.prompt_adapter.to(self.device)
        if not(self.prompt_stopper is None):
            self.prompt_stopper.to(self.device)

    def _get_inputs_dict(self, batch):
        device = self.device
        pad_token_id = self.encoder_tokenizer.pad_token_id
        source_ids, source_mask, y, ptencoder_inputs_input_ids, ptencoder_inputs_attention_mask, stopper_y = \
            batch["source_ids"], batch["source_mask"], batch["target_ids"], batch["ptencoder_inputs_input_ids"], \
            batch["ptencoder_inputs_attention_mask"], batch["stopper_y"]
        y_ids = y[:, :-1].contiguous()    # the last token (a pad_token) is removed in inputs["decoder_input_ids"]
        lm_labels = y[:, 1:].clone()      # the bos token is removed in inputs["labels"]
        lm_labels[lm_labels == pad_token_id] = -100
        lm_labels[lm_labels == self.pt_token_id] = -100
        inputs = {
            "input_ids": source_ids.to(device),
            "attention_mask": source_mask.to(device),
            "decoder_input_ids": y_ids.to(device),
            "labels": lm_labels.to(device),
            "use_casual_decoder_embeddings": self.prompt_tuning_mode or self.static_prompt_tuning_mode,    ####
        }
        if self.prompt_tuning_mode:
            ptencoder_inputs = {
                "input_ids": ptencoder_inputs_input_ids.to(device),
                "attention_mask": ptencoder_inputs_attention_mask.to(device),
            }
            stopper_y = stopper_y.to(device)
        else:
            ptencoder_inputs = None
            stopper_y = None

        return inputs, ptencoder_inputs, stopper_y

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = Seq2SeqArgs()
        args.load(input_dir)
        return args

    def get_named_parameters(self):
        return [n for n, p in self.model.named_parameters()]
