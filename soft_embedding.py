import numpy as np
import torch
import torch.nn as nn

class SoftEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 pt_id: int,
                 n_tokens: int,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = False,
                 load_from_path=None):
        """appends learned embedding
        Args:
            wte (nn.Embedding): original transformer word embedding
            pt_id: the prompt token id.
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.pt_id = pt_id
        self.n_tokens = n_tokens
        if not(load_from_path is None):
            self.learned_embedding = torch.load(load_from_path)
        else:
            self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                  n_tokens,
                                                                                  random_range,
                                                                                  initialize_from_vocab))

    def initialize_embedding(self,
                             wte: nn.Embedding,
                             n_tokens: int = 10,
                             random_range: float = 0.5,
                             initialize_from_vocab: bool = False,
                             ):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            # np.random.seed(0)
            tokens_ids = np.random.choice(np.arange(self.wte.weight.shape[0]), n_tokens, replace=False)
            return self.wte.weight[tokens_ids].clone().detach()  # TODO: change initialization scheme (include specific words?)
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specific embedding
        """

        if tokens.shape[1] == 1:   # this is a very SPECIAL DESIGN. Not recommended in general.
            return self.wte(tokens)

        assert tokens[0, 0] != self.pt_id
        assert tokens[0, 1] == self.pt_id
        # <s> before <pt>
        learned_embeddings_repeated = self.learned_embedding.repeat(tokens.size(0), 1, 1)
        assert learned_embeddings_repeated.shape[0] == tokens.shape[0]     # [bs, seqlen, embedsize]
        assert learned_embeddings_repeated.shape[1] == self.n_tokens
        assert tokens.shape[1] > self.n_tokens
        input_embedding_left = self.wte(tokens[:, :1])
        input_embedding_right = self.wte(tokens[:, self.n_tokens+1:])
        return_embeds = torch.cat([input_embedding_left, learned_embeddings_repeated, input_embedding_right], 1)
        return return_embeds


class LEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 pt_id: int,
                 n_tokens: int,
                 ):
        """appends learned embedding
        Args:
            wte (nn.Embedding): original transformer word embedding
            pt_id: the prompt token id.
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
        """
        super(LEmbedding, self).__init__()
        self.wte = wte
        self.pt_id = pt_id
        self.n_tokens = n_tokens
        self.learned_embedding = None

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specific embedding
        """

        if tokens.shape[1] == 1:   # this is a very SPECIAL DESIGN. Not recommended in general.
            return self.wte(tokens)

        assert tokens[0, 0] != self.pt_id
        assert tokens[0, 1] == self.pt_id
        # <s> before <pt>
        assert not (self.learned_embedding is None)
        assert self.learned_embedding.shape[0] == tokens.shape[0]     # [bs, seqlen, embedsize]
        assert self.learned_embedding.shape[1] == self.n_tokens
        assert tokens.shape[1] > self.n_tokens
        input_embedding_left = self.wte(tokens[:, :1])
        input_embedding_right = self.wte(tokens[:, self.n_tokens+1:])
        return_embeds = torch.cat([input_embedding_left, self.learned_embedding, input_embedding_right], 1)
        return return_embeds
