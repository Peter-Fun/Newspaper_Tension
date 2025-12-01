import os
import re
import torch
from Newspaper_Tension.dataset_bert import GDELT_BERT
from Newspaper_Tension.model_bert import BERT
import torch
import uuid
import datetime
import transformers
import tqdm
import common.logging as logging
import common.io as io
import shutil
import argparse
from copy import copy

from common.logging import setup_logging, getlogger

def open_checkpoint_file(path: str):
    """
    Return the contents of the checkpoint file.
    """
    contents = torch.load(path)
    return contents['epoch'], contents['model_state_dict'], contents['optimizer_state_dict'], contents['loss'], contents["acc"]

def load_checkpoint(model, optimizer, ckpt_folder, epoch):
    ckpt_files = io.find_files_in(ckpt_folder)
    ckpt_file = ckpt_files[epoch]
    state_epoch, model_state, optimizer_state,loss, acc = open_checkpoint_file(ckpt_file)
    optimizer.load_state_dict(optimizer_state)
    start_epoch = state_epoch
    model.load_state_dict(model_state)
    return start_epoch, model, optimizer, loss, acc

if __name__=="__main__":
    run_name = "20240615_175636_7212708a-1898-40b3-b3ae-46a470d1b97f"
    run_folder = os.path.join("runs", run_name)
    chpt_folder = os.path.join(run_folder, "checkpoints")
    test_folder = os.path.join(run_folder, "tests")
    shutil.rmtree(test_folder)
    os.mkdir(test_folder)

    setup_logging(output_dir=test_folder)
    logger = getlogger("ISEF_Training")

    parser = argparse.ArgumentParser(description="do a test run on the checkpoints")
    parser.add_argument("top", type=int, help="determine the success of the model by predicted within the top _")
    args = parser.parse_args()

    model = BERT().cuda()
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

    gdelt_test = GDELT_BERT(tokenizer=tokenizer, selected_factor = "EventRootCode", split="test")
    batch_size = 16
    test_loader = torch.utils.data.DataLoader(
        gdelt_test, batch_size, shuffle=True, num_workers=8,
    )
    criterion = torch.nn.CrossEntropyLoss()

    start = datetime.datetime.now()
    num_epochs = 42

    logger.info(f"Testing on {len(gdelt_test)} samples.")
    for epoch in range(41,num_epochs):
        start_epoch, model, optimizer, loss, acc = load_checkpoint(model, optimizer, chpt_folder, epoch)
        model.eval()
        logger.info(f"{loss}, {acc}")
        cum_train_loss = 0
        correct = 0
        i = 0
        predicted = [0] * 20
        actual = [0] * 20
        row = [0] * 20
        matrix = [[[0 for i in range(20)] for j in range(20)] for k in range(args.top)]

        predicted = [predicted[:]]
        actual = [actual[:]]
        for k in range(args.top-1):
            predicted.append(predicted[-1][:])
            actual.append(actual[-1][:])
        for sample_batch in tqdm.tqdm(test_loader, disable=True):
            i += 1
            # Split apart and prepare the sample batch
            ids = sample_batch['ids'].type(torch.LongTensor).cuda()
            token_type_ids = sample_batch['token_type_ids'].type(torch.LongTensor).cuda()
            mask = sample_batch['mask'].cuda()
            label = sample_batch['target']
            label = label.type(torch.LongTensor).cuda()
            batch_acc = 0

            logits = model(ids=ids, token_type_ids=token_type_ids, mask=mask)
            loss = criterion(logits, label)

            allpreds = torch.argsort(logits.cpu(), 1, True)

            for top_num in range(args.top):
                preds = allpreds[:, top_num]
                batch_loss = loss.detach().cpu().item()
                cum_train_loss += batch_loss * ids.shape[0]
                num_correct = (preds.detach().cpu() == label.detach().cpu()).sum().item()
                for j in range(20):
                    predicted[top_num][j] += (preds.detach().cpu() == j).sum().item()
                    actual[top_num][j] += (label.detach().cpu() == j).sum().item()
                for j in range(20):
                    for k in range(20):
                        matrix[top_num][j][k] += torch.logical_and(preds.detach().cpu() == j,label.detach().cpu() == k).sum().item()
                batch_acc += num_correct / label.shape[0]
                correct += num_correct
            if i % 5 == 0:
                logger.info(f"Test Iter: {i}/{len(test_loader)} -- loss: {batch_loss:.3f} -- accuracy: {batch_acc:.4f}")

        avg_loss = cum_train_loss / len(gdelt_test)
        acc = correct / len(gdelt_test)
        elapsed = (datetime.datetime.now() - start).seconds
        remaining = (num_epochs)/(epoch+1)*elapsed - elapsed
        h, m, s = int(remaining//3600), int((remaining%3600)//60), int(remaining%60)
        logger.info(f"ETA: {h}:{m}:{s:.2f} -- Epoch: {epoch+1}/{num_epochs} -- loss: {avg_loss:.3f} -- accuracy: {acc:.4f}")
        for top_num in range(args.top):
            logger.info(f"Top {top_num+1} Matrix: {matrix[top_num]}")
            logger.info(f"Top {top_num+1} Actual: {actual[top_num]}")
            logger.info(f"Top {top_num+1} Predicted: {predicted[top_num]}")
