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
    parser.add_argument("trainpath", type=str, help="Path to the training log file to plot")
    parser.add_argument("testpath", type=str, help="Path to the testing log file to plot")
    parser.add_argument("-a", "--smoothing-alpha", default=0.0, type=float, help="Exponential smoothing alpha value")
    args = parser.parse_args()

    with open(args.trainpath + "/stdout.log", "r") as buffer:
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
        elif "Epoch:" in line:
            a, b, c, d = line.split(" -- ")
            ep, tot_eps = b.split(" ")[-1].split("/")
            curr_epoch = int(ep)
            times.append(curr_epoch)
            accs.append(float(d.split(" ")[-1].strip()))
    


    with open(args.testpath + "/stdout.log", "r") as buffer:
        lines = buffer.readlines()

    testtimes = []
    testaccs = []
    testlosses = []
    testcurr_epoch = 0
    total = 0
    for line in lines:
        if "Test Iter:" in line:
            a, b, c = line.split(" -- ")
            it, num_its = a.split(" ")[-1].split("/")
            time = testcurr_epoch + int(it) / int(num_its)
            _, loss = b.split(" ")
            _, acc = c.split(" ")
            total += float(acc) * 16
        elif "Epoch:" in line:
            total = 0
            a, b, c, d = line.split(" -- ")
            ep, tot_eps = b.split(" ")[-1].split("/")
            testcurr_epoch = int(ep)
            testtimes.append(int(testcurr_epoch))
            testaccs.append(float(d.split(" ")[-1].strip()))
    print(accs)
    print(testaccs)
    print(losses)
    if args.smoothing_alpha:
        accs = smooth_list(accs, args.smoothing_alpha)
        losses = smooth_list(losses, args.smoothing_alpha)
        testaccs = smooth_list(testaccs, args.smoothing_alpha)
        testlosses = smooth_list(testlosses, args.smoothing_alpha)

    plt.plot(times[:-2], accs[:-2], label = "training")
    plt.plot(testtimes[:-2], testaccs[:-2], label = "testing")
    plt.title("Accuracy Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Acc.")
    plt.legend()
    plt.savefig(args.trainpath + "/both_acc.png")
    plt.clf()
