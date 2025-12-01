import argparse
import matplotlib.pyplot as plt


def smooth_list(X, alpha: float = 0.9) -> list:
    """
    Apply exponential smoothing to the list values.
    """
    out = []
    for i in range(len(X)):
        S_i = X[i] if i == 0 else (1-alpha)*out[-1] + alpha*X[i]
        out.append(S_i)
    return out


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="plot a log file")
    parser.add_argument("path", type=str, help="Path to the log file to plot")
    parser.add_argument("-a", "--smoothing-alpha", default=0.0, type=float, help="Exponential smoothing alpha value")
    parser.add_argument("--plot-acc", action="store_true", help="plot the accuracy over time")
    parser.add_argument("--plot-loss", action="store_true", help="plot the loss over time")
    args = parser.parse_args()

    with open(args.path + "/stdout.log", "r") as buffer:
        lines = buffer.readlines()

    times = []
    accs = []
    losses = []
    curr_epoch = 0
    for line in lines:
        if "Train Iter:" in line:
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
    
    if args.smoothing_alpha:
        accs = smooth_list(accs, args.smoothing_alpha)
        losses = smooth_list(losses, args.smoothing_alpha)

    if args.plot_acc:
        plt.plot(times, accs)
        plt.title("Accuracy Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Acc.")
        plt.savefig(args.path + "/_acc.png")
        plt.clf()

    if args.plot_loss:
        plt.title("Loss Over Time")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(times, losses)
        plt.savefig(args.path + "/_loss.png")