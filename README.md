## Sentencepiece Tokenizer Wrapper for PLDR-LLM with KV-cache and G-cache

This repository implements a wrapper code for generating a Sentencepiece Vocabulary and Tokenizer model from RefinedWeb dataset using pytorch/torchtune framework. The tokenizers generated with this wrapper script are used in the research article: [PLDR-LLMs Learn A Generalizable Tensor Operator That Can Replace Its Own Deep Neural Net At Inference](https://arxiv.org/abs/2502.13502).

More information on Sentencepiece tokenizer can be found in articles:
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226) 
- [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/abs/1804.10959). 

The Git repo for Sentencepiece module is at [https://github.com/google/sentencepiece](https://github.com/google/sentencepiece).

#### Key features

- Builds a Sentencepiece tokenizer model from dataset.
- Optimized for preprocessing RefinedWeb dataset to generate a tokenizer model.
- The tokenizer implements reserved tokens "[PAD]", "[UNK]", "[START]", "[END]", "[SEP]", "[CLS]" into vocabulary for research purposes.
- For PLDR-LLM, an "[END]" token at the end of each sentence and "[PAD]" token for padding is used.

#### Setting Hyperparameters for Sentencepiece Model Training:
Some of the hyperparameters for Sentencepiece can be provided through a parameter dictionary. proto_name is used for creating file names for sentencepiece model, tokenizer and preprocessed dataset files. If preprocessed text file is available, its path can be provided through "data_as_text_file" key.

```python
import pretrain_make_sentencepiece_tokenizer_pt as ptmspt

proto_name=os.path.abspath("/file/path/to/tokenizer/model")
sp_tokenizer_params={"lowercase":False, 
                     "vocabulary_size":32000, 
                     "model_type":"unigram", 
                     "proto_output_file":proto_name, 
                     "num_threads":None,
                     "input_sentence_size":5000000,
                     "max_sentence_length":int(4192*30),
                     "shuffle_input_sentence":True,
                     "minloglevel":0,
                     "data_as_text_file":None
                    }
```

Below features of the tokenizer model are predefined in the wrapper module:
```
--pad_id=0 --unk_id=1 
--bos_id=2 --eos_id=3
--user_defined_symbols=[SEP],[CLS] 
--split_digits=True 
--byte_fallback=True 
--hard_vocab_limit=False
--unk_piece=[UNK] --bos_piece=[START] 
--eos_piece=[END] --pad_piece=[PAD]
```

#### Training a Sentencepiece Model and Tokenizer:

The tokenizer and a sentencepiece model can be trained as follows.

```python
refinedweb_sp_tokenizer = ptmspt.sentencepiece_src_tokenizer(
                 dataset_file="tiiuae/falcon-refinedweb",
                 dataset_name="falcon-refinedweb",
                 split_style="index",
                 train_intvl=[0, 2000000],
                 model_path = f"{proto_name}-tokenizer",
                 load_tokenizer_model=False,
                 make_tokenizer=True,
                 sp_tokenizer_params=sp_tokenizer_params,
                 shuffle_files=True,
                 shuffle_seed=1234
                 )
```

#### Loading The Tokenizer

```python

model_name = "/path/to/tokenizer/model"

#load tokenizer model
sp_tokenizer=ptmspt.load_sentencepiece_tokenizer(model_name)

```

#### Encoding and Decoding with The Tokenizer

Below methods are accessible through the tokenizer to encode text and decode tokens, view reserved tokens or get the vocabulary size.

```python
sp_tokenizer.encode("Hello world! How are you?", add_bos= False, add_eos = True, trim_leading_whitespace = False, prefix = None)

sp_tokenizer.decode([596, 291, 282, 319, 1506, 282, 435, 262, 3])

#print several attributes in the tokenizers class
print(sp_tokenizer.vocab_size, sp_tokenizer.bos_id, sp_tokenizer.eos_id, sp_tokenizer.pad_id, sp_tokenizer.spm_model)

```
