import time
import os
from os import listdir
from os.path import isfile, join
from tempfile import gettempdir
import json
import codecs
from argparse import ArgumentParser
from pandas import array
import tqdm
import pandas as pd
from io import StringIO

from clearml import Task, Dataset

def model_training():

    PROJECT_NAME = "fgET"
    TASK_NAME = "model testing (1 mil)"

    Task.force_requirements_env_freeze(force=True, requirements_file='requirements.txt')
    Task.add_requirements("torch")
    task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
    # task.set_base_docker("nvcr.io/nvidia/pytorch:20.08-py3")
    task.set_base_docker("nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04")

    args = {'project_name':PROJECT_NAME,'task_name':TASK_NAME}
    task.connect(args)

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--svd')
    arg_parser.add_argument('--lr', type=float, default=1e-5)
    arg_parser.add_argument('--max_epoch', type=int, default=40)
    arg_parser.add_argument('--batch_size', type=int, default=32)
    arg_parser.add_argument('--elmo_dataset_project', type=str, default='datasets/multimodal')
    arg_parser.add_argument('--elmo_dataset_name', type=str, default='elmo weights')
    arg_parser.add_argument('--elmo_option_file', type=str, default='elmo_5.5B_options.json')
    arg_parser.add_argument('--elmo_weights_file', type=str, default='elmo_5.5B_weights.hdf5')
    arg_parser.add_argument('--elmo_dropout', type=float, default=.5)
    arg_parser.add_argument('--repr_dropout', type=float, default=.2)
    arg_parser.add_argument('--dist_dropout', type=float, default=.2)
    arg_parser.add_argument('--gpu', type=bool, default = True)
    arg_parser.add_argument('--device', type=int, default=0)
    arg_parser.add_argument('--weight_decay', type=float, default=0.01)
    arg_parser.add_argument('--latent_size', type=int, default=0)
    arg_parser.add_argument('--train', type=bool, default = True)
    arg_parser.add_argument('--test', type=bool, default = True)
    arg_parser.add_argument('--labels_dataset_project', type=str, default='datasets/multimodal')
    arg_parser.add_argument('--labels_dataset_name', type=str, default='fgET HAnDS 1k manual verified preprocessed')
    arg_parser.add_argument('--labels_file_name', type=str, default='ner_tags.json')
    arg_parser.add_argument('--fgETdata_dataset_project', type=str, default='datasets/multimodal')
    arg_parser.add_argument('--fgETdata_dataset_name', type=str, default='fgET HAnDS 1k manual verified preprocessed')
    arg_parser.add_argument('--train_file_name', type=str, default='train.parquet')
    arg_parser.add_argument('--val_file_name', type=str, default='validation.parquet')
    arg_parser.add_argument('--test_file_name', type=str, default='test.parquet')
    arg_parser.add_argument('--tokens_field', type=str, default='tokens')
    arg_parser.add_argument('--entities_field', type=str, default='fine_grained_entities')
    arg_parser.add_argument('--sentence_field', type=str, default='text')
    arg_parser.add_argument('--results_dataset_project', type=str, default='datasets/multimodal')
    arg_parser.add_argument('--results_dataset_name', type=str, default='fgET 1mil test results')
    arg_parser.add_argument('--train_from_checkpoint', type=bool, default=False)
    arg_parser.add_argument('--test_from_checkpoint', type=bool, default=True)
    arg_parser.add_argument('--model_checkpoint_project', type=str, default='datasets/multimodal')
    arg_parser.add_argument('--model_checkpoint_dataset_name', type=str, default='fgET 1mil trained')
    arg_parser.add_argument('--model_checkpoint_file_name', type=str, default='best_mac.mdl')

    args = arg_parser.parse_args()
    task.connect(vars(args),name='General')

    task.execute_remotely()

    logger = task.get_logger()

    # ============= imports =============
    from collections import defaultdict

    import torch
    from torch.utils.data import DataLoader

    from model.fgET_model import fgET
    from model.fgET_data import FetDataset
    from model.fgET_preprocessor import PreProcessor

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    results_dataset_name = args.results_dataset_name + ' ' + timestamp
    num_worker = 4

    print("Loading labels dictionary from clearML {} dataset from file {}".format(args.labels_dataset_name,args.labels_file_name))
    labels_file_path = get_clearml_file_path(args.labels_dataset_project,args.labels_dataset_name,args.labels_file_name)

    print("Loading elmo embeddings from clearML {} dataset".format(args.elmo_dataset_name))
    elmo_option = get_clearml_file_path(args.elmo_dataset_project,args.elmo_dataset_name,args.elmo_option_file)
    elmo_weight = get_clearml_file_path(args.elmo_dataset_project,args.elmo_dataset_name,args.elmo_weights_file)

    with open(labels_file_path) as json_file:
        labels_strtoidx = json.load(json_file)

    labels_idxtostr = {i: s for s, i in labels_strtoidx.items()}
    label_size = len(labels_strtoidx)
    print('Label size: {}'.format(len(labels_strtoidx)))

    if args.test_from_checkpoint:
        print("------------ Performing model testing!!!! ------------")

        test_file_path = get_clearml_file_path(args.fgETdata_dataset_project,args.fgETdata_dataset_name,args.test_file_name)
        preprocessor = PreProcessor(labels_strtoidx,
                                    elmo_option=elmo_option,
                                    elmo_weight=elmo_weight,
                                    elmo_dropout=args.elmo_dropout)
        test_set = FetDataset(preprocessor,test_file_path,args.tokens_field,args.entities_field,args.sentence_field,labels_strtoidx,args.gpu)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,collate_fn=preprocessor.batch_process,num_workers=num_worker)

        # Set GPU device
        gpu = torch.cuda.is_available() and args.gpu
        if gpu:
            torch.cuda.set_device(args.device)

        # Build model
        model = fgET(label_size,
                    elmo_dim = preprocessor.elmo_dim,
                    repr_dropout=args.repr_dropout,
                    dist_dropout=args.dist_dropout,
                    latent_size=args.latent_size,
                    svd=args.svd
                    )
        if gpu:
            model.cuda()

        total_step = len(test_loader)
        optimizer = model.configure_optimizers(args.weight_decay,args.lr,total_step)

        model_file_path = get_clearml_file_path(args.model_checkpoint_project,args.model_checkpoint_dataset_name,args.model_checkpoint_file_name)
        print("Retrieving model checkpoint from {}".format(model_file_path))
        checkpoint = torch.load(model_file_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for name, param in model.named_parameters():print(name, param)

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
            'vocab': {'label': labels_strtoidx}
        }

        best_scores = {
            'best_acc_val': 0, 'best_mac_val': 0, 'best_mic_val': 0,
            'best_acc_test': 0, 'best_mac_test': 0, 'best_mic_test': 0
        }

        results,best_scores = run_test(test_loader,model,logger,best_scores,args.gpu)
        collated_results = {'results':[]}
        for gold, pred, men_id,mention,sentence,score in zip(results['gold'],results['pred'],results['ids'],results['mentions'],results['sentence'],results['scores']):
                arranged_results = dict()
                gold_labels = [labels_idxtostr[i] for i, l in enumerate(gold) if l]
                pred_labels = [labels_idxtostr[i] for i, l in enumerate(pred) if l]
                arranged_results['mention_id'] = men_id
                arranged_results['mention'] = mention
                arranged_results['sentence'] = sentence
                arranged_results['gold'] = gold_labels
                arranged_results['predictions'] = pred_labels
                arranged_results['scores'] = [score[i]for i, l in enumerate(pred) if l]
                collated_results['results'].append(arranged_results)

        # dataset = Dataset.create(
        #     dataset_project=args.results_dataset_project, dataset_name=args.results_dataset_name
        # )

        dataset = create_dataset(args.results_dataset_project,args.results_dataset_name)

        with codecs.open(os.path.join(gettempdir(), 'results.json'), mode='w', encoding='utf-8',
                    errors='ignore') as fp:
            json.dump(collated_results, fp=fp, ensure_ascii=False, indent = 4)
        
        training_data = collated_results['results']
        training_records = {}

        for i in range(0,len(training_data)):
            training_records[str(i)] = training_data[i]

        json_object = json.dumps(training_records, indent = 4)
        df = pd.read_json(StringIO(json_object), orient ='index')
        df.to_csv(os.path.join(gettempdir(), 'results.csv'),index=False)
        logger.report_table(title='results',series='pandas DataFrame',iteration=0,table_plot=df)

        files = [f for f in listdir(gettempdir()) if isfile(join(gettempdir(), f)) and (f.endswith('.json') or f.endswith('.mdl') or f.endswith('.csv'))]

        for file in files:
            dataset.add_files(os.path.join(gettempdir(), file))

        dataset.upload(output_url='s3://experiment-logging/multimodal')
        dataset.finalize()

        return

    # Load data sets
    print('Loading data sets')

    train_file_path = get_clearml_file_path(args.fgETdata_dataset_project,args.fgETdata_dataset_name,args.train_file_name)
    val_file_path = get_clearml_file_path(args.fgETdata_dataset_project,args.fgETdata_dataset_name,args.val_file_name)
    test_file_path = get_clearml_file_path(args.fgETdata_dataset_project,args.fgETdata_dataset_name,args.test_file_name)

    preprocessor = PreProcessor(labels_strtoidx,
                                elmo_option=elmo_option,
                                elmo_weight=elmo_weight,
                                elmo_dropout=args.elmo_dropout)
    
    train_set = FetDataset(preprocessor,train_file_path,args.tokens_field,args.entities_field,args.sentence_field,labels_strtoidx,args.gpu)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,collate_fn=preprocessor.batch_process,num_workers=num_worker)
    val_set = FetDataset(preprocessor,val_file_path,args.tokens_field,args.entities_field,args.sentence_field,labels_strtoidx,args.gpu)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,collate_fn=preprocessor.batch_process,num_workers=num_worker)
    if args.test:
        test_set = FetDataset(preprocessor,test_file_path,args.tokens_field,args.entities_field,args.sentence_field,labels_strtoidx,args.gpu)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,collate_fn=preprocessor.batch_process,num_workers=num_worker)


    # Set GPU device
    gpu = torch.cuda.is_available() and args.gpu
    if gpu:
        torch.cuda.set_device(args.device)

    # Build model
    model = fgET(label_size,
                elmo_dim = preprocessor.elmo_dim,
                repr_dropout=args.repr_dropout,
                dist_dropout=args.dist_dropout,
                latent_size=args.latent_size,
                svd=args.svd
                )
    if gpu:
        model.cuda()

    torchscript = torch.jit.script(model)

    total_step = args.max_epoch * len(train_loader)
    optimizer = model.configure_optimizers(args.weight_decay,args.lr,total_step)

    if args.train_from_checkpoint:

        model_file_path = get_clearml_file_path(args.model_checkpoint_project,args.model_checkpoint_dataset_name,args.model_checkpoint_file_name)
        print("Retrieving model checkpoint from {}".format(model_file_path))
        checkpoint = torch.load(model_file_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        for name, param in model.named_parameters():print(name, param)

        #try freezing
        for name, param in model.named_parameters():
            if param.requires_grad and ('latent_scalar' in name or 'output_linear' in name or 'latent_to_label' in name or 'feat_to_latent' in name):
                 param.requires_grad = True
            else:
                param.requires_grad = False

        for name, param in model.named_parameters():print(name, param)

    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': vars(args),
        'vocab': {'label': labels_strtoidx}
    }

    best_scores = {
        'best_acc_val': 0, 'best_mac_val': 0, 'best_mic_val': 0,
        'best_acc_test': 0, 'best_mac_test': 0, 'best_mic_test': 0
    }

    if args.train:
        model,state,best_scores = run_training(train_loader,val_loader,model,optimizer,args.max_epoch,logger,state,best_scores,args.gpu)
    if args.test:
       results,best_scores = run_test(test_loader,model,logger,best_scores,args.gpu)
       collated_results = {'results':[]}
       for gold, pred, men_id,mention,sentence,score in zip(results['gold'],results['pred'],results['ids'],results['mentions'],results['sentence'],results['scores']):
            arranged_results = dict()
            gold_labels = [labels_idxtostr[i] for i, l in enumerate(gold) if l]
            pred_labels = [labels_idxtostr[i] for i, l in enumerate(pred) if l]
            arranged_results['mention_id'] = men_id
            arranged_results['mention'] = mention
            arranged_results['sentence'] = sentence
            arranged_results['gold'] = gold_labels
            arranged_results['predictions'] = pred_labels
            arranged_results['scores'] = [score[i]for i, l in enumerate(pred) if l]
            collated_results['results'].append(arranged_results)

    # dataset = Dataset.create(
    #     dataset_project=args.results_dataset_project, dataset_name=args.results_dataset_name
    # )

    dataset = create_dataset(args.results_dataset_project,args.results_dataset_name)

    if args.test:
        with codecs.open(os.path.join(gettempdir(), 'results.json'), mode='w', encoding='utf-8',
                    errors='ignore') as fp:
            json.dump(collated_results, fp=fp, ensure_ascii=False, indent = 4)
        
        training_data = collated_results['results']
        training_records = {}

        for i in range(0,len(training_data)):
            training_records[str(i)] = training_data[i]

        json_object = json.dumps(training_records, indent = 4)
        df = pd.read_json(StringIO(json_object), orient ='index')
        df.to_csv(os.path.join(gettempdir(), 'results.csv'),index=False)
        logger.report_table(title='results',series='pandas DataFrame',iteration=0,table_plot=df)

    files = [f for f in listdir(gettempdir()) if isfile(join(gettempdir(), f)) and (f.endswith('.json') or f.endswith('.mdl') or f.endswith('.csv'))]

    for file in files:
        dataset.add_files(os.path.join(gettempdir(), file))

    dataset.upload(output_url='s3://experiment-logging/multimodal')
    dataset.finalize()



def run_training(train_loader,validation_loader,model,optimizer,epochs,logger,state,best_scores,gpu=False):

    from collections import defaultdict
    import torch
    from model.fgET_scorer import calculate_metrics

    for epoch in range(epochs):
        print('-' * 20, 'Epoch {}'.format(epoch), '-' * 20)
        start_time = time.time()

        epoch_loss = []
        val_loss = []
        progress = tqdm.tqdm(total=len(train_loader)+len(validation_loader), mininterval=1,
                            desc='Epoch: {}'.format(epoch))

        print("Training step...")
        for batch in train_loader:

            elmo_embeddings, labels, men_masks, ctx_masks, dists, gathers, men_ids, mentions,sentences = batch

            # if epoch == 0:
            #     for i in range(0,len(mentions)):
            #         print("Mention: ",mentions[i],"sentence: ",sentences[i])


            if gpu:
                elmo_embeddings = elmo_embeddings.to(device='cuda')
                labels = torch.cuda.FloatTensor(labels)
                men_masks = torch.cuda.FloatTensor(men_masks)
                ctx_masks = torch.cuda.FloatTensor(ctx_masks)
                gathers = torch.cuda.LongTensor(gathers)
                dists = torch.cuda.FloatTensor(dists)

            else:
                labels = torch.FloatTensor(labels)
                men_masks = torch.FloatFloatTensorTensor(men_masks)
                ctx_masks = torch.LongTensor(ctx_masks)
                gathers = torch.LongTensor(gathers)
                dists = torch.FloatTensor(dists)

            progress.update(1)
            optimizer.zero_grad()
            loss = model.forward(elmo_embeddings, labels, men_masks, ctx_masks, dists,
                                 gathers)

            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())

        avg_train_loss = sum(epoch_loss)/len(epoch_loss)
        if logger is not None:
            logger.report_scalar(title='Train',series='Loss', value=avg_train_loss,iteration=epoch)

        print("Validation step...")

        model.eval()
        
        results = defaultdict(list)
        with torch.no_grad():
            for batch in validation_loader:

                elmo_embeddings, labels, men_masks, ctx_masks, dists, gathers, men_ids, mentions,sentences = batch

                if gpu:
                    elmo_embeddings = elmo_embeddings.to(device='cuda')
                    labels = torch.cuda.FloatTensor(labels)
                    men_masks = torch.cuda.FloatTensor(men_masks)
                    ctx_masks = torch.cuda.FloatTensor(ctx_masks)
                    gathers = torch.cuda.LongTensor(gathers)
                    dists = torch.cuda.FloatTensor(dists)

                else:
                    labels = torch.FloatTensor(labels)
                    men_masks = torch.FloatFloatTensorTensor(men_masks)
                    ctx_masks = torch.LongTensor(ctx_masks)
                    gathers = torch.LongTensor(gathers)
                    dists = torch.FloatTensor(dists)


                progress.update(1)

                preds,scores = model.predict(elmo_embeddings, men_masks, ctx_masks, dists, gathers)
                results['gold'].extend(labels.int().data.tolist())
                results['pred'].extend(preds.int().data.tolist())
                results['ids'].extend(men_ids)

                loss = model.forward(elmo_embeddings, labels, men_masks, ctx_masks, dists,
                                    gathers)

                val_loss.append(loss.item())

        model.train()

        avg_val_loss = sum(val_loss)/len(val_loss)
        if logger is not None:
            logger.report_scalar(title='Validation Loss',series='Loss', value=avg_val_loss,iteration=epoch)

        metrics = calculate_metrics(results['gold'], results['pred'])
        print('---------- Validation set ----------')
        print('Strict accuracy: {:.2f}'.format(metrics.accuracy))
        if logger is not None:
            logger.report_scalar(title='Validation Accuracy',series='Accuracy', value=metrics.accuracy,iteration=epoch)
            logger.report_scalar(title='Validation Accuracy',series='Macro Fscore', value=metrics.macro_fscore,iteration=epoch)
            logger.report_scalar(title='Validation Accuracy',series='Micro Fscore', value=metrics.micro_fscore,iteration=epoch)
        print('Macro: P: {:.2f}, R: {:.2f}, F:{:.2f}'.format(
            metrics.macro_prec,
            metrics.macro_rec,
            metrics.macro_fscore))
        print('Micro: P: {:.2f}, R: {:.2f}, F:{:.2f}'.format(
            metrics.micro_prec,
            metrics.micro_rec,
            metrics.micro_fscore))
        # Save model
        if metrics.accuracy > best_scores['best_acc_val']:
            best_scores['best_acc_val'] = metrics.accuracy
        if metrics.macro_fscore > best_scores['best_mac_val']:
            best_scores['best_mac_val'] = metrics.macro_fscore
            print('Saving new best macro F1 model')
            state['model']= model.state_dict()
            state['optimizer']= optimizer.state_dict()
            torch.save(state, os.path.join(gettempdir(), 'best_mac.mdl'))
        if metrics.micro_fscore > best_scores['best_mic_val']:
            best_scores['best_mic_val'] = metrics.micro_fscore
            print('Saving new best micro F1 model')
            state['model']= model.state_dict()
            state['optimizer']= optimizer.state_dict()
            torch.save(state, os.path.join(gettempdir(), 'best_mic.mdl'))

        progress.close()


    return model,state,best_scores

def run_test(test_loader,model,logger,best_scores,gpu=False):
    from collections import defaultdict
    import torch
    from model.fgET_scorer import calculate_metrics

    progress = tqdm.tqdm(total=len(test_loader), mininterval=1,
                        desc='Test')

    results = defaultdict(list)
    with torch.no_grad():
        for batch in test_loader:

            elmo_embeddings, labels, men_masks, ctx_masks, dists, gathers, men_ids, mentions,sentences = batch

            if gpu:
                elmo_embeddings = elmo_embeddings.to(device='cuda')
                labels = torch.cuda.FloatTensor(labels)
                men_masks = torch.cuda.FloatTensor(men_masks)
                ctx_masks = torch.cuda.FloatTensor(ctx_masks)
                gathers = torch.cuda.LongTensor(gathers)
                dists = torch.cuda.FloatTensor(dists)

            else:
                labels = torch.FloatTensor(labels)
                men_masks = torch.FloatFloatTensorTensor(men_masks)
                ctx_masks = torch.LongTensor(ctx_masks)
                gathers = torch.LongTensor(gathers)
                dists = torch.FloatTensor(dists)


            progress.update(1)

            preds,scores = model.predict(elmo_embeddings, men_masks, ctx_masks, dists, gathers)
            results['gold'].extend(labels.int().data.tolist())
            results['pred'].extend(preds.int().data.tolist())
            results['scores'].extend(scores.tolist())
            results['ids'].extend(men_ids)
            results['mentions'].extend(mentions)
            results['sentence'].extend(sentences)

    metrics = calculate_metrics(results['gold'], results['pred'])
    print('---------- Test set ----------')
    print('Strict accuracy: {:.2f}'.format(metrics.accuracy))
    if logger is not None:
        logger.report_scalar(title='Test',series='Accuracy', value=metrics.accuracy,iteration=1)
    print('Macro: P: {:.2f}, R: {:.2f}, F:{:.2f}'.format(
        metrics.macro_prec,
        metrics.macro_rec,
        metrics.macro_fscore))
    print('Micro: P: {:.2f}, R: {:.2f}, F:{:.2f}'.format(
        metrics.micro_prec,
        metrics.micro_rec,
        metrics.micro_fscore))
    
    best_scores['best_acc_test'] = metrics.accuracy
    best_scores['best_mac_test'] = metrics.macro_fscore
    best_scores['best_mic_test'] = metrics.micro_fscore

    for k, v in best_scores.items():
        print('{}: {:.2f}'.format(k.replace('_', ' '), v))
    progress.close()

    return results,best_scores


def get_clearml_file_path(dataset_project,dataset_name,file_name):

    print("Getting files from: ",dataset_project,dataset_name,file_name)

    # get uploaded dataset from clearML
    dataset_dict = Dataset.list_datasets(
        dataset_project=dataset_project, partial_name=dataset_name, only_completed=False
    )

    datasets_obj = [
        Dataset.get(dataset_id=dataset_dict["id"]) for dataset_dict in dataset_dict
    ]

    # reverse list due to child-parent dependency, and get the first dataset_obj
    dataset_obj = datasets_obj[::-1][0]
    
    folder = dataset_obj.get_local_copy()

    file = [file for file in dataset_obj.list_files() if file==file_name][0]

    file_path = folder + "/" + file

    return file_path


def create_dataset(dataset_project, dataset_name):
    parent_dataset = _get_last_child_dataset(dataset_project, dataset_name)
    if parent_dataset:
        print("create child")
        parent_dataset.finalize()
        child_dataset = Dataset.create(
            dataset_name, dataset_project, parent_datasets=[parent_dataset]
        )
        return child_dataset
    else:
        print("create parent")
        dataset = Dataset.create(dataset_name, dataset_project)
        return dataset


def _get_last_child_dataset(dataset_project, dataset_name):
    datasets_dict = Dataset.list_datasets(
        dataset_project=dataset_project, partial_name=dataset_name, only_completed=False
    )
    if datasets_dict:
        datasets_dict_latest = datasets_dict[-1]
        return Dataset.get(dataset_id=datasets_dict_latest["id"])


if __name__ == '__main__':
    from torch.multiprocessing import Pool, Process, set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    model_training()