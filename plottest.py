import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# python plottest.py --plot-acc --plot-loss "runs/20240318_232455_1726701e-8c51-4913-9137-b73753d42439/tests/stdout.log"
def smooth_list(X, alpha: float = 0.9) -> list:
    """
    Apply exponential smoothing to the list values.
    """
    out = []
    for i in range(len(X)):
        S_i = X[i] if i == 0 else (1-alpha)*out[-1] + alpha*X[i]
        out.append(S_i)
    return out

def plot_confusion_matrix(df_confusion, path, epoch, place, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.colorbar()
    tick_marks = np.arange(20)
    plt.xticks(tick_marks, list(range(1,21)))
    plt.yticks(tick_marks, list(range(1,21)))
    plt.ylabel("Predicted")
    plt.xlabel("Actual")
    plt.savefig(path + "/top" + str(place) + "_test_confusion_matrix" + str(epoch+1) + ".png")



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="plot a log file")
    parser.add_argument("path", type=str, help="Path to the log file to plot")
    parser.add_argument("-a", "--smoothing-alpha", default=0.0, type=float, help="Exponential smoothing alpha value")
    parser.add_argument("--plot-acc", action="store_true", help="plot the accuracy over time")
    parser.add_argument("--plot-loss", action="store_true", help="plot the loss over time")
    parser.add_argument("--confusion-matrix", action = "store_true", help="plot the confusion matrix")
    parser.add_argument("--top", type = int, help = "the top _ prediction used for accuracy")
    args = parser.parse_args()

    with open(args.path + "/stdout.log", "r") as buffer:
        lines = buffer.readlines()
    times = []
    accs = []
    losses = []
    actual = []
    predicted = []
    matrices = []
    curr_epoch = 0
    for line in lines:
        if "Test Iter:" in line:
            a, b, c = line.split(" -- ")
            it, num_its = a.split(" ")[-1].split("/")
            time = curr_epoch + int(it) / int(num_its)
            _, loss = b.split(" ")
            _, acc = c.split(" ")
            losses.append(float(loss))
            accs.append(float(acc))
            times.append(time)
        elif "Epoch:" in line:
            a, b, c, d = line.split(" -- ")
            ep, tot_eps = b.split(" ")[-1].split("/")
            curr_epoch = int(ep)
        elif "Actual:" in line:
            actual.append([int(i) for i in line.split("[")[4][:-2].split(", ")])
        elif "Predicted:" in line:
            predicted.append([int(i) for i in line.split("[")[4][:-2].split(", ")])
        elif "Matrix:" in line:
            rows = line.split(": ")[1][0:-1]
            rows = rows.split('[')[2:]
            rows[-1] = rows[1][:-1] + " "
            for j in range(len(rows)):
                rows[j] = [int(k) for k in rows[j][:-3].split(", ")]
            matrices.append(rows)
    
    if args.smoothing_alpha:
        accs = smooth_list(accs, args.smoothing_alpha)
        losses = smooth_list(losses, args.smoothing_alpha)

    if args.plot_acc:
        plt.plot(times, accs)
        plt.title("Accuracy Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Acc.")
        plt.savefig(args.path + "/test_acc.png")
        plt.clf()

    if args.plot_loss:
        plt.title("Loss Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(times, losses)
        plt.savefig(args.path + "/test_loss.png")

    if args.confusion_matrix:
        for i in range(int(len(matrices)/ args.top)):
            for j in range(args.top):
                y_actu = pd.Series(actual[i], name='Actual')
                y_pred = pd.Series(predicted[i], name='Predicted')
                df_confusion = pd.crosstab(y_actu, y_pred)
                plot_confusion_matrix(matrices[i*args.top + j],args.path,i,j+1)