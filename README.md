# BERT_Dynamic_Quantization

### Dependencies
```
pip install sklearn
pip install transformers
yes y | pip uninstall torch tochvision
yes y | pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html
```

### GLUE Data
```
python download_glue_data.py --data_dir='glue_data' --tasks='MRPC'
```
Microsoft Research Paraphrase Corpus (MRPC) is a corpus consists of 5,801 sentence pairs collected from newswire articles. Each pair is labelled if it is a paraphrase or not by human annotators. The whole set is divided into a training subset (4,076 sentence pairs of which 2,753 are paraphrases) and a test subset (1,725 pairs of which 1,147 are paraphrases).

### HuggingFace BERT Optimization
Fine-tune the deep bi-directional representations:
```
export GLUE_DIR=./glue_data
export TASK_NAME=MRPC
export OUT_DIR=./$TASK_NAME/
python ./run_glue.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --save_steps 100000 \
    --output_dir $OUT_DIR
```

## References

[1] J.Devlin, M. Chang, K. Lee and K. Toutanova, BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018).

[2] HuggingFace Transformers.

[3] O. Zafrir, G. Boudoukh, P. Izsak, and M. Wasserblat (2019). Q8BERT: Quantized 8bit BERT.
