import covid
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from matplotlib.font_manager import FontProperties

fpmi = {
    "Week": [dt.date(2023, 5, 8), dt.date(2023, 5, 15), dt.date(2023, 5, 22), dt.date(2023, 5, 29)],
    "東京都": [2.40, 3.53, 3.96, 5.29],
    "滋賀県": [1.82, 2.07, 1.77, 2.47],
    "鳥取県": [2.69, 3.24, 2.86, 4.24],
    "全国": [2.63, 3.55, 3.63, 4.55]
}

def ex05(pref, epoch0, ndays, xlim, xticks, ylim, yticks, pngfile, log=False, rotation=28, pad=-2):
    weekly1, weekly2 = covid.readXLSXfp()
    fig = plt.figure(figsize=(6, 4))
    ax = plt.axes([0.09, 0.11, 0.89, 0.83])
    xlim, xticks = [epoch0, epoch0 + dt.timedelta(days=ndays)], [epoch0 + dt.timedelta(days=i) for i in range(7, ndays, 4 * 7)]
    if log:
        ax.set_yscale("log")
    p = covid.prefecture[pref]
    covid.drawFrame(pref + "定点医療機関 定点あたりの報告数", ax, xlim, xticks, ylim, yticks, rotation=rotation, pad=pad)
    ax.bar([d[0] for d in weekly2["Week"]], weekly2[pref], label=f"定点あたりの報告数", align="center", width=7.0, alpha=0.3, color="blue")
    ax.bar(fpmi["Week"], np.array(fpmi[pref]), align="center", width=7.0, alpha=0.3, color="orange")
    ax.legend(loc="upper right")
    plt.xticks(rotation=rotation)
    plt.savefig(pngfile, dpi=600)

if __name__ == "__main__":
    epoch0, ndays = dt.datetime(2022, 1, 10), 82 * 7
    xlim, xticks = [epoch0, epoch0 + dt.timedelta(days=ndays)], [epoch0 + dt.timedelta(days=i) for i in range(7, ndays, 4 * 7)]
    pref, ylim, yticks = "滋賀県", [0.2, 200], [1, 10, 100]
    ex05(pref, epoch0, ndays, xlim, xticks, ylim, yticks, "ex05.png", log=True)
