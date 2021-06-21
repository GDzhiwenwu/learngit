# This project uses the structure of MUSE (https://github.com/facebookresearch/MUSE)

import torch
from torch import nn
import torch.nn.functional as F

from .utils import load_embeddings, normalize_embeddings




class NonLinearMapper(nn.Module):
    """
    Single hidden layer FFN with tanh activation
    """
    def __init__(self, params, mapper_AB=True):
        super(NonLinearMapper, self).__init__()
        self.emb_dim_A = params.emb_dim_autoenc_A if mapper_AB else params.emb_dim_autoenc_B
        self.emb_dim_B = params.emb_dim_autoenc_B if mapper_AB else params.emb_dim_autoenc_A
        self.hidden_dim = params.mapper_hidden_dim

        self.lin_layer1 = nn.Linear(self.emb_dim_A, self.hidden_dim)
        self.lin_layer2 = nn.Linear(self.hidden_dim, self.emb_dim_B)

        if getattr(params, 'map_id_init', True):
            nn.init.zeros_(self.lin_layer1.weight.data).fill_diagonal_(1)
            nn.init.zeros_(self.lin_layer2.weight.data).fill_diagonal_(1)

        self.dropuout_layer = nn.Dropout(params.mapper_dropout)
        self.nonlinearity = nn.Tanh()

class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.emb_dim = params.emb_dim_autoenc
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)  

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)
class Discriminator_Align(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.emb_dim = params.emb_dim_autoenc*2
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)

class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.emb_dim = params.emb_dim
        self.bottleneck_dim = params.emb_dim_autoenc #400
        self.l_relu = params.l_relu #
        self.encoder = nn.Linear(self.emb_dim, self.bottleneck_dim)
        self.leakyRelu = nn.LeakyReLU(0.2)
    def forward(self, x):
        x = self.encoder(x)
        if self.l_relu == 1:
            x = self.leakyRelu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()

        self.emb_dim = params.emb_dim
        self.bottleneck_dim = params.emb_dim_autoenc
        self.decoder = nn.Linear(self.bottleneck_dim, self.emb_dim)
        self.leakyRelu = nn.LeakyReLU(0.2)
    def forward(self, x):
        x = self.decoder(x)
        return x

def build_model(params, with_dis):
    """
    Build all components of the model.
    """
    # source embeddings
    src_dico, _src_emb = load_embeddings(params, source=True) 
    params.src_dico = src_dico
    src_emb = nn.Embedding(len(src_dico), params.emb_dim, sparse=True)
    src_emb.weight.data.copy_(_src_emb)

    # target embeddings
    if params.tgt_lang:
        tgt_dico, _tgt_emb = load_embeddings(params, source=False)
        params.tgt_dico = tgt_dico
        tgt_emb = nn.Embedding(len(tgt_dico), params.emb_dim, sparse=True)
        tgt_emb.weight.data.copy_(_tgt_emb)
    else:
        tgt_emb = None

    # mapping
    # mapping_G = nn.Linear(params.emb_dim_autoenc, params.emb_dim_autoenc, bias=False)
    # mapping_F = nn.Linear(params.emb_dim_autoenc, params.emb_dim_autoenc, bias=False)

    mapping_G = NonLinearMapper(params, mapper_AB=True)
    mapping_F = NonLinearMapper(params, mapper_AB=False)

    if getattr(params, 'map_id_init', True):
        mapping_G.weight.data.copy_(torch.diag(torch.ones(params.emb_dim_autoenc)))
        mapping_F.weight.data.copy_(torch.diag(torch.ones(params.emb_dim_autoenc)))
    
    # discriminator
    discriminator_A = Discriminator(params) if with_dis else None
    discriminator_B = Discriminator(params) if with_dis else None


    discriminator_A_Align = Discriminator_Align(params) if with_dis else None
    discriminator_B_Align = Discriminator_Align(params) if with_dis else None

    # autoencoder
    encoder_A = Encoder(params)
    decoder_A = Decoder(params)
    encoder_B = Encoder(params)
    decoder_B = Decoder(params)

    # cuda
    if params.cuda:
        src_emb.cuda()
        if params.tgt_lang:
            tgt_emb.cuda()
        mapping_G.cuda()
        mapping_F.cuda()
        if with_dis:
            discriminator_A.cuda()
            discriminator_B.cuda()
            discriminator_A_Align.cuda()
            discriminator_B_Align.cuda()
        encoder_A.cuda()
        decoder_A.cuda()
        encoder_B.cuda()
        decoder_B.cuda()

    # normalize embeddings
    params.src_mean = normalize_embeddings(src_emb.weight.data, params.normalize_embeddings)
    if params.tgt_lang:
        params.tgt_mean = normalize_embeddings(tgt_emb.weight.data, params.normalize_embeddings)

    return src_emb, tgt_emb, mapping_G, mapping_F, discriminator_A, discriminator_B, encoder_A, decoder_A, encoder_B, decoder_B,discriminator_A_Align,discriminator_B_Align
