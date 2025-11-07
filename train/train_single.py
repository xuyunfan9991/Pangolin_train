import sys
from model import *
from torch import randperm
#from utils_torch import progress_bar


def crossent(y_pred, y_true):
    """Categorical cross entropy for sequence outputs."""
    return - torch.mean(
        y_true[:, 0, :] * torch.log(y_pred[:, 0, :] + 1e-10)
        + y_true[:, 1, :] * torch.log(y_pred[:, 1, :] + 1e-10)
    )

assert torch.cuda.is_available()

N_GPUS = 1
N_CORES = 6
BATCH_SIZE = 12
model_num = int(sys.argv[1])
TISSUE = int(sys.argv[3])
torch.backends.cudnn.benchmark = True

def random_split(dataset, lengths):
    total_length = sum(lengths)
    if total_length != len(dataset):
        raise ValueError(
            "Sum of input lengths does not equal the length of the dataset"
        )

    indices = randperm(total_length).tolist()
    splits = []
    offset = 0
    for length in lengths:
        next_offset = offset + length
        splits.append(data.Subset(dataset, indices[offset:next_offset]))
        offset = next_offset

    return tuple(splits)

ds = H5Dataset(sys.argv[4])
val = round(0.1*len(ds))
train = len(ds)-val
train_ds, val_ds = data.random_split(ds, (train,val))

train_dl = data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CORES, pin_memory=True)
val_dl = data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=N_CORES, pin_memory=True)

model = Pangolin(L, W, AR)
if torch.cuda.device_count() > 1:
    print("Using %s gpus" % torch.cuda.device_count())
    model = nn.DataParallel(model)
model.cuda()
model.load_state_dict(torch.load(sys.argv[2]))

bce = nn.BCELoss()


def loss(y_pred, y_true):
    y_pred = torch.split(y_pred, [2,1,2,1,2,1,2,1], dim=1)
    y_true = torch.split(y_true, [2,1,2,1,2,1,2,1], dim=1)

    if TISSUE % 2 == 0:
        loss = crossent(y_pred[TISSUE], y_true[TISSUE])
    else:
        loss = bce(
            y_pred[TISSUE][y_true[TISSUE] >= 0],
            y_true[TISSUE][y_true[TISSUE] >= 0],
        )

    return loss

criterion = loss
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
T_0 = 4
T_mult = 2
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
iters = len(train_dl)
flog_final = open("log.%s.%s.txt" % (model_num, TISSUE), 'w')


def train_epoch(epoch):
    model.train()
    train_loss = 0.0
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl, start=1):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        batch_loss = criterion(outputs, targets)
        train_loss += float(batch_loss)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        scheduler.step(epoch + (batch_idx - 1) / max(1, iters))

        num_batches = batch_idx

    return train_loss / max(1, num_batches)


def evaluate():
    model.eval()
    test_loss = 0.0
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(val_dl, start=1):
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            outputs = model(inputs)
            batch_loss = criterion(outputs, targets)
        test_loss += float(batch_loss)
        num_batches = batch_idx

    return test_loss / max(1, num_batches)


for epoch in range(0, 4):
    train_loss = train_epoch(epoch)
    test_loss = evaluate()
    print(epoch, train_loss, test_loss, file=flog_final)
    flog_final.flush()
    torch.save(model.state_dict(), "models/final.%s.%s.%s" % (model_num, TISSUE, epoch))



