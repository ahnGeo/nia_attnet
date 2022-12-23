import argparse


def val_loss_and_acc(file_path) :
    with open(file_path, 'r') as f :
        data = f.readlines()
        
    val_loss = []
    val_acc = []
    
    for line in data :
        if "| validation loss" in line :
            val_loss.append(line.split()[-1])
        elif "| validation acc" in line :
            val_acc.append(line.split()[-1])
    
    with open("train_results/val_loss_{}.txt".format(file_path.split("/")[-1].split(".")[0]), 'w') as f :
        f.write("\n".join(val_loss))
        
    with open("train_results/val_acc_{}.txt".format(file_path.split("/")[-1].split(".")[0]), 'w') as f :
        f.write("\n".join(val_acc))
        
if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str)
    args = parser.parse_args()
    
    val_loss_and_acc(args.file_path)