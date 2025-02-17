''''
Run module to train tokenizer model and make vocabulary from pretrained dataset for Large Language Model from Power Law Decoder Representations
with KV-cache and G-cache.
'''

import time

import datasets as hfds
import sentencepiece_tokenizer_pt as spt
from torchtune.modules.tokenizers import SentencePieceBaseTokenizer


class sentencepiece_src_tokenizer:
    '''
    Creates a a custom sentencepiece tokenizer object and vocabulary from dataset.
    '''
    def __init__(self,
                 dataset_file="tiiuae/falcon-refinedweb",
                 dataset_name='falcon-refinedweb',
                 split_style="index",
                 train_intvl=None,
                 model_path = "./refinedweb_pretrain_en_tokenizer",
                 load_tokenizer_model=False,
                 make_tokenizer=True,
                 sp_tokenizer_params=None,
                 shuffle_dataset=False,
                 shuffle_seed=1234
                 ):
        '''
        Arguments:
            dataset_file: Location of dataset on disk to load.
            dataset_name: A name used for running a pre-defined preprocessing procedure for dataset.
            split_style: index or percent to use as split train_intvl.
            train_intvl: A tuple for start and end indices/percent for dataset split. None loads all data.
            model_path: Sentencepiece model path location to save under or load from.
            load_tokenizer_model: If True loads tokenizer model at model_path path. 
                                  Dataset is not used to create vocabulary and tokenizer. Default is False.
            make_tokenizer: If True tokenizer is created from vocabulary and saved at model_path. 
                            If False only vocabulary is created from dataset. Default is True.
            sp_tokenizer_params: Parameter dict for sentencepiece tokenizer.
            shuffle_dataset: Shuffle the dataset after loading. Default: False
            shuffle_seed: seed for shuffling the dataset.
        '''

        self.load_tokenizer_model=load_tokenizer_model
        self.model_path = model_path
        self.make_tokenizer=make_tokenizer

        if self.load_tokenizer_model:
            #load tokenizer model only from model_path.
            print("TOKENIZER INITIALIZED FROM SAVED MODEL")
            self.tokenizer=load_sentencepiece_tokenizer(self.model_path)
        else:
            if sp_tokenizer_params:
                self.sp_tokenizer_params= sp_tokenizer_params
            else:
                self.sp_tokenizer_params={"lowercase": False, 
                                        "vocabulary_size": 32000, 
                                        "model_type": "unigram", 
                                        "proto_output_file": "my_sp_model",
                                        "num_threads": None,
                                        "input_sentence_size": None,
                                        "max_sentence_length":None,
                                        "shuffle_input_sentence": True,
                                        "minloglevel": None,
                                        "use_iterator": False,
                                        "data_as_text_file": None
                                        }
        
            self.src_proto_path=self.sp_tokenizer_params["proto_output_file"]
            self.src_proto_path=f"{self.src_proto_path}.model"

            # create model and vocabulary generator objects
            self.src_vocab_obj = spt.gen_sp_proto(sp_tokenizer_params)
            if self.sp_tokenizer_params["data_as_text_file"] is None:
                print("LOADING DATASET")
                if train_intvl:
                    start_ind, end_ind=train_intvl
                    if split_style=="percent":
                        examples = hfds.load_dataset(dataset_file,
                                                     split=[f'train[{start_ind}%:{end_ind}%]'])
                        if shuffle_dataset:
                            print("SHUFFLING DATASET")
                            examples[0]=examples[0].shuffle(seed=shuffle_seed)
                    if split_style=="index":
                        examples = hfds.load_dataset(dataset_file,
                                                     split=[f'train[{start_ind}:{end_ind}]'])
                        if shuffle_dataset:
                            print("SHUFFLING DATASET")
                            examples[0]=examples[0].shuffle(seed=shuffle_seed)
                    else:
                        print(f"Warning: Invalid split style specified: {split_style}. Choose from percent or index")
                else:
                    #load all data
                    examples = hfds.load_dataset(dataset_file, split=["train"])
                    if shuffle_dataset:
                        print("SHUFFLING DATASET")
                        examples[0]=examples[0].shuffle(seed=shuffle_seed)
                
                if dataset_name in ['falcon-refinedweb']:
                    print("FULL DATASET STRUCTURE")
                    print(examples)
                    print("REDUCING DATASET TO SAMPLES ONLY")
                    examples=examples[0]['content']
                    #print a few samples from dataset
                    print(f"Printing a few examples from preprocessed dataset {dataset_name}")
                    for i in examples[:4]:
                        print(i)
                else:
                    print(f"Warning: Invalid dataset name {dataset_name}. hf preprocessing not done.")

                self.train_examples = examples
            else:
                print(f"SKIPPED LOADING DATASET. DATASET AS TEXT FILE PROVIDED AT:{self.sp_tokenizer_params['data_as_text_file']}")

            print("GENERATING SPM MODEL AND VOCABULARY")
            self.src_make_proto()
            print("MODEL AND VOCABULARY DONE")

    def src_make_proto(self):
        '''
        Method to create sentencepiece proto file and vocabulary from dataset.
        Returns: path to the proto file
        '''

        if self.load_tokenizer_model:
            print(f"Tokenizer model is loaded from {self.model_path}")
            self.src_proto_path= None
        else:
            if self.src_proto_path:
                print(f"Creating proto file for tokenizer at {self.src_proto_path}")
                start=time.time()
                if self.sp_tokenizer_params["data_as_text_file"] is None:
                    self.src_vocab_obj.generate_sp_proto(self.train_examples)
                else:
                    self.src_vocab_obj.generate_sp_proto()
                print(f"Proto file done in {time.time() - start:.2f} s")

        return self.src_proto_path

def load_sentencepiece_tokenizer(proto_file_path):
    tokenizer=SentencePieceBaseTokenizer(proto_file_path)
    print([item for item in dir(tokenizer) if not item.startswith('_')])
    print("Tokenizer Loaded.")
    return tokenizer
