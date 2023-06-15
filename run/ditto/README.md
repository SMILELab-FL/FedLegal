> ditto算法需要记录并保持每个客户端的本地模型，而master分支采用`GPU Balance`的策略进行联邦训练加速，无法保证同一个客户端的本地模型保留在同一个进程的内存中以便使用。在后续的框架优化中，我们将考虑跨进程通信来传递本地模型信息。
对此，我们建议用户切换到`ditto`分支进行ditto算法的运行，该分支采用原始的`Fixed GPU Map`为客户端分配固定的GPU，从而保障本地模型的完整记录。

---
> The ditto algorithm needs to record and maintain the local model of each client, while the master branch adopts the strategy of `GPU Balance` for federated training acceleration, which cannot guarantee that the local model of the same client is kept in the memory of the same process for use. In subsequent framework optimizations, we will consider cross-process communication to transfer local model information.
In this regard, we recommend that users switch to the `ditto` branch to run the ditto algorithm. This branch uses the original `Fixed GPU Map` to assign a fixed GPU to the client, so as to ensure the complete record of the local model.