'''
Module for creating a sentencepiece tokenizer model and vocabulary and define a custom tokenizer class as wrapper
for the tokenizer model.
'''

import numpy as np
import random
import sentencepiece as spm
import time


# print("Setting random seeds")
# random.seed(1234)
# np.random.seed(1234)
# spm.set_random_generator_seed(1234)

class gen_sp_proto():
    '''
    This class is a wrapper for generating sentencepiece model and vocabulary.
    '''
    def __init__(self, sp_tokenizer_params=None):
        '''
        Arguments:
            sp_tokenizer_params: dictionary of parameters for sentencepiece tokenizer model trainer
        '''


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
                                      "data_as_text_file": None
                                      }
            

        self.reserved_tokens= ["[PAD]", "[UNK]", "[START]", "[END]", "[SEP]", "[CLS]"]

    def generate_sp_proto(self, ds=None):
        '''
        Generates sentencepiecemodel and vocabulary from dataset
        Arguments:
            ds: A dataset containing list of text.
        Returns:
            A sentencepiece model and vocabulary for the dataset.
        '''

        proto_output_file=self.sp_tokenizer_params["proto_output_file"]
        vocabulary_size=self.sp_tokenizer_params["vocabulary_size"]
        model_type=self.sp_tokenizer_params["model_type"]
        lowercase=self.sp_tokenizer_params["lowercase"]
        num_threads=self.sp_tokenizer_params["num_threads"]
        input_sentence_size=self.sp_tokenizer_params["input_sentence_size"]
        max_sentence_length=self.sp_tokenizer_params["max_sentence_length"]
        shuffle_input_sentence=self.sp_tokenizer_params["shuffle_input_sentence"]
        minloglevel=self.sp_tokenizer_params["minloglevel"]
        data_as_text_file=self.sp_tokenizer_params["data_as_text_file"]

        if data_as_text_file is None:
            print("WRITING DATASET TO TEXT FILE")
            start=time.time()
            with open(f"{proto_output_file}-data-raw.txt", "w") as dstxtfile:
                for t in ds:
                    dstxtfile.write(t)
            #remove blank lines between sentences and write to a final file.
            with open(f"{proto_output_file}-data-raw.txt", "r"
                        ) as dsreadfile, open(f"{proto_output_file}-data.txt", "w") as dswritefile:
                for line in dsreadfile:
                    if line.strip():
                        dswritefile.write(line)
                
            data_as_text_file=f"{proto_output_file}-data.txt"
            print(f"WRITING DATASET TO TEXT FILE FINISHED IN {time.time()-start:.2f}s AT: {data_as_text_file}")
        else:
            print(f"USING PROVIDED DATA TEXT FILE AS INPUT: {data_as_text_file}")
        
        normalization_rule_name="nmt_nfkc_cf" if lowercase else "nmt_nfkc"
        max_sentence_length=int(max_sentence_length) if max_sentence_length is not None else 4192
        num_threads=num_threads if num_threads is not None else 16
        input_sentence_size=input_sentence_size if input_sentence_size is not None else 0
        shuffle_input_sentence=shuffle_input_sentence if shuffle_input_sentence is not None else True 
        minloglevel=minloglevel if minloglevel is not None else 0
        
        #define the command line string
        sp_cmd_line=f"--input={data_as_text_file} --model_type={model_type} --model_prefix={proto_output_file} --vocab_size={vocabulary_size} \
                        --normalization_rule_name={normalization_rule_name} --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 \
                        --user_defined_symbols=[SEP],[CLS] --split_digits=True --byte_fallback=True --hard_vocab_limit=False \
                        --unk_piece=[UNK] --bos_piece=[START] --eos_piece=[END] --pad_piece=[PAD] --max_sentence_length={max_sentence_length} \
                        --num_threads={num_threads} --input_sentence_size={input_sentence_size} --shuffle_input_sentence={shuffle_input_sentence} \
                        --minloglevel={minloglevel}"
        print(sp_cmd_line)

        #give some time for interfaces to post outputs
        print("STARTING SENTENCEPIECE TRAINING")
        time.sleep(5)

        spm.SentencePieceTrainer.train(sp_cmd_line)

        return f"{proto_output_file}.model"
    





