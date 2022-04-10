from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator


# Modified from
# https://github.com/open-mmlab/mmdetection/blob/master/tools/analysis_tools/confusion_matrix.py#L146
def plot_confusion_matrix(confusion_matrix,
                          labels,
                          is_normzlized=False,
                          title='Confusion Matrix',
                          color_theme='plasma'):

    if is_normzlized:
        per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
        confusion_matrix = \
            confusion_matrix.astype(np.float32) / per_label_sums * 100

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(0.5 * num_classes, 0.5 * num_classes * 0.8), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(confusion_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    title_font = {'weight': 'bold', 'size': 8}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 6}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confution matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            if is_normzlized:
                ax.text(
                    j,
                    i,
                    '{}%'.format(
                        int(confusion_matrix[
                                i,
                                j]) if not np.isnan(confusion_matrix[i, j]) else -1),
                    ha='center',
                    va='center',
                    color='w',
                    size=7)
            else:
                ax.text(
                    j,
                    i,
                    '{}'.format(
                        int(confusion_matrix[
                                i,
                                j]) if not np.isnan(confusion_matrix[i, j]) else -1),
                    ha='center',
                    va='center',
                    color='w',
                    size=7)

    ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    class_name = ['cat', 'dog', 'sheep', 'pig']
    len_y_true = 120
    y_true = np.random.randint(0, 4, 120)
    y_pred = np.random.randint(0, 4, 120)
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_name, is_normzlized=True, title='Normalized Confusion Matrix')
