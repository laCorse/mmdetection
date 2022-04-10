import numpy as np
import matplotlib.pyplot as plt


def calc_PR_curve1(pred, label):
    threshold = np.sort(pred)[::-1]
    precision = []
    recall = []

    def get_tp_pp(y, proba, threshold):
        """Return the number of true positives."""
        pred = np.where(proba >= threshold, 1, 0)
        tp = np.sum((y == 1) & (pred == 1))
        positive_predictions = np.sum(pred == 1)
        return tp, positive_predictions

    for i in range(len(threshold)):
        tp, positive_predictions = get_tp_pp(label, pred, threshold[i])
        precision.append(tp / positive_predictions)
        recall.append(tp / sum(label))

    return precision, recall


def calc_PR_curve(pred, label):
    threshold = np.sort(pred)[::-1]
    label = label[pred.argsort()[::-1]]
    precision = []
    recall = []
    tp = 0
    fp = 0
    ap = 0  # 平均精度
    for i in range(len(threshold)):
        if label[i] == 1:
            tp += 1
            recall.append(tp / len(label))
            precision.append(tp / (tp + fp))
            ap += (recall[i] - recall[i - 1]) * precision[i]  # 近似曲线下面积
        else:
            fp += 1
            recall.append(tp / len(label))
            precision.append(tp / (tp + fp))

    return precision, recall, ap


def plot_pr_curve(precision, recall):
    plt.plot(recall, precision, '-bv', drawstyle="steps")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title('Precision Recall Curve')
    # plt.show()


if __name__ == '__main__':
    preds = np.array([0.6, 0.8, 0.2, 0.8, 0.5, 0.7])
    labels = np.array([0, 1, 0, 1, 1, 1])

    precision, recall, ap = calc_PR_curve(preds, labels)
    print('ap:',ap)
    print(precision)
    print(recall)
    plot_pr_curve(precision, recall)

    from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay)
    precision, recall, thresholds = precision_recall_curve(labels, preds)
    print(precision)
    print(recall)
    print(thresholds)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot(marker='v')
    plt.show()
