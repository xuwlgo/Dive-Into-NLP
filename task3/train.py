import os
import torch
import torch.nn as nn
from model import ESIM
from utils import train, validate
from data_process import LCQMC_Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def data_loader(max_length=50):
    """data_loader
    加载数据（训练集、验证集）,并进行数据清洗
    Arguments:
        max_length:语句“归一化”长度
    Returns:
        train_loader:训练集迭代器
        dev_loader:验证集迭代器
    """
    train_file = 'data/atec_nlp_sim_train_all.csv'
    vocab_file = 'data/vocab.txt'
    dev_file = 'data/dev.csv'

    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    train_data = LCQMC_Dataset(train_file, vocab_file, max_length)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=256, num_workers=4)
    print("\t* Loading validation data...")
    dev_data = LCQMC_Dataset(dev_file, vocab_file, max_length)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=256, num_workers=4)

    print(len(train_data),len(dev_data))   # 92477,10000
    p_l,p_len,h_l,h_len,label = train_data[0]  # getitem()
    print(p_l.shape,p_len,'\n',h_l.shape,h_len,'\n',label)

    return train_loader,dev_loader


def train_model(device,model,num_epochs,PATH,
         lr=0.0005,
         patience=5,
         max_grad_norm=10.0):

    train_loader,dev_loader = data_loader()

    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    # 自适应学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.85, patience=0)
    best_score = 0.0
    start_epoch = 0
    writer = SummaryWriter()

    # 加载预训练模型
    if os.path.exists(PATH) is not True:
        criterion = nn.CrossEntropyLoss()
    else:
        checkpoint = torch.load(PATH)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        criterion = checkpoint['criterion']
    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy, auc = validate(model, dev_loader, criterion)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}".format(valid_loss,
                                                                                (valid_accuracy * 100),auc))
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training ESIM model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, num_epochs):
        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer,
                                                       criterion, epoch, max_grad_norm)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        writer.add_scalar('train/train_loss', epoch_loss, epoch)
        writer.add_scalar('train/train_acc', epoch_accuracy, epoch)

        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, epoch_auc = validate(model, dev_loader, criterion)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100), epoch_auc))
        writer.add_scalar('test/test_loss',epoch_loss,epoch)
        writer.add_scalar('test/test_acc',epoch_accuracy,epoch)

        # 自适应学习率调整
        scheduler.step(epoch_accuracy)

        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score},
                       os.path.join(os.path.dirname(PATH), "best.tar"))
        if epoch % 5 == 0:
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "optimizer": optimizer.state_dict(),
                        "criterion":criterion},PATH)
        writer.close()
        # 模型效果连续5个epoch未提升，训练提前终止
        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == "__main__":
    # data_loader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = './check_point/check_point.tar'
    num_epochs = 50

    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    model = ESIM(hidden_size=300, dropout=0.2,
                 num_labels=2, device=device).to(device)

    # -------------------- Preparation for training  ------------------- #
    train_model(device,model,num_epochs,model_path)

