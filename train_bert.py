from Newspaper_Tension.dataset_bert import GDELT_BERT
from Newspaper_Tension.model_bert import BERT
import os
import torch
import uuid
import datetime
import transformers
import tqdm

from common.logging import setup_logging, getlogger

run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())
run_folder = os.path.join("runs", run_name)
chpt_folder = os.path.join(run_folder, "checkpoints")
os.mkdir(run_folder)
os.mkdir(chpt_folder)
setup_logging(output_dir=run_folder)
logger = getlogger("ISEF_Training")

"""
TODO (Peter):
1. Use the summaries instead.
2. Try running on full dataset.
3. Change into Classification Task.
"""

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
gdelt_train = GDELT_BERT(tokenizer=tokenizer, selected_factor = "EventRootCode", split="train") #https://mind-node.net/cameo-event-codes/
gdelt_test = GDELT_BERT(tokenizer=tokenizer, selected_factor = "EventRootCode", split="test")

batch_size = 32

train_loader = torch.utils.data.DataLoader(
    gdelt_train, batch_size, shuffle=True, num_workers=8,
)
test_loader = torch.utils.data.DataLoader(
    gdelt_test, batch_size, shuffle=True, num_workers=8,
)

model = BERT().cuda()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


criterion = torch.nn.CrossEntropyLoss()
model.train()
num_epochs = 50
start = datetime.datetime.now()
logger.info(f"Training on {len(gdelt_train)} samples.")
for epoch in range(50):
    cum_train_loss = 0
    correct = 0
    i = 0
    model.train(True)
    for sample_batch in tqdm.tqdm(train_loader, disable=True):
        i += 1
        # Split apart and prepare the sample batch
        ids = sample_batch['ids'].type(torch.LongTensor).cuda()
        token_type_ids = sample_batch['token_type_ids'].type(torch.LongTensor).cuda()
        mask = sample_batch['mask'].cuda()
        label = sample_batch['target']
        label = label.type(torch.LongTensor).cuda()

        optimizer.zero_grad()
        logits = model(ids=ids, token_type_ids=token_type_ids, mask=mask)
        loss = criterion(logits, label)
        preds = torch.argmax(logits.cpu(), 1)

        # Backwards Pass
        loss.backward()
        optimizer.step()

        batch_loss = loss.detach().cpu().item()
        cum_train_loss += batch_loss * ids.shape[0]
        num_correct = (preds.detach().cpu() == label.detach().cpu()).sum().item()
        batch_acc = num_correct / label.shape[0]
        correct += num_correct
        if i % 5 == 0:
            logger.info(f"Train Iter: {i}/{len(train_loader)} -- loss: {batch_loss:.3f} -- accuracy: {batch_acc:.4f}")
            
    predicted = preds[0].item()
    actual = label[0].item()
    avg_loss = cum_train_loss / len(gdelt_train)
    acc = correct / len(gdelt_train)
    elapsed = (datetime.datetime.now() - start).seconds
    remaining = (num_epochs)/(epoch+1)*elapsed - elapsed
    h, m, s = int(remaining//3600), int((remaining%3600)//60), int(remaining%60)
    logger.info(f"ETA: {h}:{m}:{s:.2f} -- Epoch: {epoch+1}/{num_epochs} -- loss: {avg_loss:.3f} -- accuracy: {acc:.4f}")

    # Save the model.
    if (epoch+1) % 1 == 0:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "acc": acc,
        }, os.path.join(chpt_folder, f"checkpoint_{str(epoch+1).zfill(4)}.pth"))