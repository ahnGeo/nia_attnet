import argparse


def val_loss_and_acc(file_path) :
    with open(file_path, 'r') as f :
        data = f.readlines()
        
    val_loss = []
    val_acc = []
    train_loss = []
    
    state = 0
    for line in data :
        if "| validation loss" in line :
            val_loss.append(line.split()[-1])
        elif "| validation acc" in line :
            val_acc.append(line.split(":")[-1][1:-1])
        elif "| iteration" in line :
            if state % 5 == 0 :
                train_loss.append(line.split()[-1])
            state += 1
    
    with open("train_results/val_loss_{}.txt".format(file_path.split("/")[-1].split(".")[0].split("_")[-1]), 'w') as f :
        f.write("\n".join(val_loss))
        
    with open("train_results/val_acc_{}.txt".format(file_path.split("/")[-1].split(".")[0].split("_")[-1]), 'w') as f :
        f.write("\n".join(val_acc))
        
    with open("train_results/train_loss_{}.txt".format(file_path.split("/")[-1].split(".")[0].split("_")[-1]), 'w') as f :
        f.write("\n".join(train_loss))
        
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    args = parser.parse_args()
    
    val_loss_and_acc(args.file_path)