# -*- coding: utf-8 -*-


# +
class Config():
    model_label = 'test'

    # ELMo
    elmo_options_file = "./data/elmo/elmo_2x2048_256_2048cnn_1xhighway_options.json"
    elmo_weight_file = "./data/elmo/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
    elmo_dim = 512
    #elmo_dim = 200

    # Bert
    bert_path = './data/bert/'
    bert_dim = 768

    # glove
    vocab_size = 18766
    glove_dim = 300
    glove_file = "./data/glove/glove_300d.npy"
    word2id_file = "./data/glove/word2id.npy"

    emb_method = 'glove'  # bert/elmo/glove/
    enc_method = 'CNN'  # CNN/RNN/Transformer/mean
    hidden_size = 200
    out_size = 512
    num_labels = 2

    use_gpu = True
    seed = 2020
    gpu_id = 0
### parameters for GloVe
#     dropout = 0.5
#     epochs = 70

#     test_size = 0.2
#     lr = 1e-3
#     weight_decay = 2e-4
#     batch_size = 70


###  parameters for ELMO  
    dropout = 0.5
    epochs = 30

    test_size = 0.2
    lr = 1e-3
    weight_decay = 2e-4
    batch_size = 32
    device = "cuda:0"


# -

def parse(self, kwargs):
    '''
    user can update the default hyperparamter
    '''
    for k, v in kwargs.items():
        if not hasattr(self, k):
            raise Exception('opt has No key: {}'.format(k))
        setattr(self, k, v)

    print('*************************************************')
    print('user config:')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print("{} => {}".format(k, getattr(self, k)))

    print('*************************************************')


Config.parse = parse
opt = Config()
