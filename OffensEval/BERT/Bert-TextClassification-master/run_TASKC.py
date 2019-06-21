# coding=utf-8
from main import main
import os
from config import *

if __name__ == "__main__":

    model_name = "BertOrigin"
    data_dir = TASKC_DATA_DIR
    output_dir = "output/taskc_output/" 
    cache_dir = "output/taskc_cache"
    log_dir = "output/taskc_log/" 

    model_times = "model_1/"   # 第几次保存的模型，主要是用来获取最佳结果

    # bert-base
    bert_vocab_file = os.path.join(PRETRAIN_BERT_DIR,PRETRAIN_BERT_VOC)
    bert_model_dir = PRETRAIN_BERT_DIR

#     # bert-large
#     bert_vocab_file = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-large-uncased-vocab.txt"
#     bert_model_dir = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-large-uncased"
    
    from Processors.TASKCProcessor import TaskcProcessor

    if model_name == "BertOrigin":
        from BertOrigin import args

    elif model_name == "BertCNN":
        from BertCNN import args

    elif model_name == "BertATT":
        from BertATT import args

    elif model_name == "BertRCNN":
        from BertRCNN import args

    elif model_name == "BertCNNPlus":
        from BertCNNPlus import args

    main(args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir),
             model_times, TaskcProcessor)
        


