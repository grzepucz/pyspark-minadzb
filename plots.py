import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import seaborn as sn


def plot_corr_matrix(correlations, attr, title=''):
    loc = plticker.MultipleLocator(base=1.0)  # this locator puts ticks at regular intervals
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.set_title(title)
    ax.set_xticklabels(list(attr))
    ax.set_yticklabels(list(attr))
    cax = ax.matshow(correlations)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

    plt.show()


def draw_table(headings, data, title=''):
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')
    ax.set_title(title)
    table = ax.table(cellText=[[float("{0:.4f}".format(x)) for x in data]], colLabels=headings, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    plt.show()


def draw_histogram(xs, title):
    plt.hist(x=[float(x) for x in xs], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85, orientation=u'vertical')
    plt.xticks(rotation=45)
    plt.title(title)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()


def plot_confusion_matrix(dataset, title):
    plt.figure(figsize=(10, 7))
    sn.heatmap(dataset, annot=True)
    plt.title = title
    plt.show()
