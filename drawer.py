import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re

with open("data_source.txt") as f:
    datas = [[], [], [], [], [], []]
    for line in f.readlines():
        data = re.findall(r"\d+\.?\d*", line)
        if len(data) != 7:
            continue
        data = [eval(i) for i in data]
        datas[0].append(data[1])
        datas[1].append(data[2])
        datas[2].append(data[3])
        datas[3].append(data[4])
        datas[4].append(data[5])
        datas[5].append(data[6])

    datas = np.array(datas).T
    datas = pd.DataFrame(data=datas, columns=["task latency", "task_1 completion rate",
                                              "task_2 completion rate",
                                              "task_3 completion rate",
                                              "loss critic", "loss actor"
                                              ])
    datas = datas.rolling(window=50).mean()
    datas["completion rate"] = datas[["task_1 completion rate",
                                              "task_2 completion rate",
                                              "task_3 completion rate"]].mean(axis=1)
    # sns.lineplot(data=datas[["task_1 completion rate",
    #                                           "task_2 completion rate",
    #                                           "task_3 completion rate"]])

    # sns.lineplot(data=datas[["task latency"]])
    # sns.lineplot(data=datas[["loss critic"]])
    sns.lineplot(data=datas[["completion rate"]])
    # sns.lineplot(data=datas[["task_1 completion rate",
    #                                           "task_2 completion rate",
    #                                           "task_3 completion rate"]])
    print(datas.corr())

    # sns.scatterplot(data=datas, x="task_1 completion rate", y="task latency", legend="auto")
    # sns.scatterplot(data=datas, x="task_2 completion rate", y="task latency", legend="auto")
    # sns.scatterplot(data=datas, x="task_3 completion rate", y="task latency", legend="auto")
    # sns.scatterplot(data=datas, x="completion rate", y="task latency", legend="auto")


    plt.show()
