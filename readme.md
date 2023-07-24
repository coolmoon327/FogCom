# 使用深度强化学习算法求解的基于 Stackelberg 博弈模型的雾计算典型场景建模

## 研究背景

### 背景来源

​	本文研究内容将作为 OpenRaaS 的一部分，目的是探讨适合于 OpenRaaS 的调度算法。由于 OpenRaaS 环境本身的复杂性，以及其研究领域的受限性，我们希望在一个更加泛用、简单的环境内开展调度算法的设计工作。该应用环境可以与 OpenRaaS 的背景设计存在偏差，但一些基本要点必须与之吻合，尤其是作为核心的双决策主体问题。

### 基本要点

​	按照 OpenRaaS 的调度流程，环境设计的过程中需要满足以下特性：

1. 两个决策主体——leader + follower
   - leader 负责选择 follower，并为 follower 调配一个资源池
   - follower 负责执行具体的任务，该任务由来自它自己以及资源池中的资源来完成
2. 核心问题——partial observation
   - leader 只有局部视野，它不清楚 follower 与资源池中具体某项资源的网络链路关系
   - follower 具有更多的视野，但它不能总是向 leader 汇报
3. 性能指标——多个 QoS 值
   - 受到 follower 自身条件、follower 的决策模型、从资源池中选择的 host machine 的影响
4. 优化目标——基于这些 QoS 值的 social welfare
   - social welfare = (unity-price) + (price-cost) 是 leader 的目标
5. 决策变量——follower 能观测到的资源单价，或者资源池内容
   - price-cost 是 follower 的目标，所以可以通过 price 来控制 follower 的决策
     - 本质上是控制 follower 能够看见的每个 QoS 指标的权重，它在策略中会根据这些权重估算目标
   - follower 从资源池中选择资源，所以控制资源池内容能够控制 follower 的决策

### 问题设计

​	我们选用的模型是雾计算。其中，参与计算的节点都是边缘侧的服务设备，既可以是自愿提供算力的可信第三方设备，也可以是无线接入网中（RAN）基站上的服务器。雾计算致力于池化这些异构的用户侧设备，因此一种有效的雾计算方案是通过虚拟机或容器来利用这些异构的资源：当任务到来时，被分配的服务器将根据任务内容从云盘（另一个设备）中获取能够支持该服务的虚拟机（VM），整个过程类似数据中心对 on-rack 设备的池化。这个过程需要两类节点：计算节点与存储节点。前者就是参与雾计算的节点，负责提供边缘算力；后者既可以是边缘节点，也可以是云数据中心，负责为计算节点提供 VM。在这个模型中，我们采用资源即服务（RaaS）的方法对资源进行池化，也就是每个设备只关心自己提供的资源类型与资源量，而不关心具体提供的服务内容。因此，调度与计价都被有效简化。

​	我们提出一种基于雾计算的合作边缘计算（CEC）方案，其中边缘设备的资源是池化的，以地理空间上的小区为单位进行管理。在每个小区中存在一个 leader 节点 $l$，它负责管理与调度池化的资源。当一个任务到来时，它将被用户所在小区的基站接收，并转发给 $l$，然后 $l$ 根据任务内容在小区内选择出合适的计算和存储资源来提供服务。我们认为任务不能被分配到其他的小区去，因为过远的距离将影响边缘计算的低延迟性能。在这个方案中，参与雾计算的边缘设备因为其移动性和无线链路的不确定性，由 $l$ 动态维护所有节点间的链路信息实际上是不可取的。因此 $l$ 无法对 QoS 目标进行精确量化，完全依靠 $l$ 进行调度可能导致服务质量的下降。因此，$l$ 只选择一个 follower 节点 $f$ 作为计算节点，并选出一批合适的存储站点 $D$ 告知 $f$，之后交由 $f$ 自行决定最佳的 $d \in D$。

​	为了简化建模，我们采用标准的计算卸载任务，并根据任务完成时间来决定用户的权益（utility）。该建模中采用三个 QoS 指标：传输速度，服务延迟，与计算性能。在计算卸载的场景下，它们可以共同反映在任务从发起到完成的时间间隔上。于是，多 QoS 本质上回到了单目标：

1. **总时间 = 任务上传时间  + VM 加载时间 + 任务处理时间 + 结果下发时间**
2. 任务上传时间 = task size / 用户-计算节点带宽 + 用户-计算节点延迟
3. VM 加载时间 = VM block size / min(计算-存储节点带宽, 存储介质读取速度) + 计算-存储节点延迟
4. 任务处理时间 = task 所需 cycles 数 / 计算节点的频率
5. 结果下发时间 = result size / 用户-计算节点带宽 + 用户-计算节点延迟

​	其中，因为 VM 是挂载到 $f$ 上的，任务到来时 $f$ 不需要下载全部的 VM，而是“即用即取”。它会以数据块（block）的形式依次缓存接下来将使用的内容，因此只有在请求第一个数据块时需要等待其加载。对于 $f$ 来说，无论任务上传还是 VM 加载，都是占用的下载带宽，因此二者无法简单地同时进行，在此设置为顺序加载。

​	由于本文探讨的是 leader 与 follower 博弈与合作的问题，因此不强调 $l$ 选择 $f$ 的策略。在此，我们采用一个最朴素的贪心策略进行 $f$ 的选择：从用户所在小区内选择一个能够立即执行该任务的、根据局部观测信息（partial observation）有最大目标估计值的计算节点。

## 基础模型

### 系统设计

​	当编号为 $u_i$ 的用户设备发起编号为 $i$ 的任务时，任务信息 $B_i=\{i,t_i,s_i,w_i,sid_i,b_i(*)\}$ 将被递交给其所在区域的 leader 节点 $L(u)$。其中 $t_i$ 是任务到达时间，$s_i$ 是本地上传数据大小，$w_i$ 是完成该任务需要的 CPU 周期数，$sid_i$ 表示该任务的类型，对应一个能够提供该类服务的 $vm_i$，$b_i(*)$ 是期望价值函数。函数 $R(sid_i)=vm_i$ 用于检索该任务对应的虚拟机编号（若无特别说明，本文默认所有编号从 1 开始）。值得注意的是，$b_i(*)$ 并不是定价函数 $v_i(*)$，它代表的是用户心中的估价，在此我们采用线性函数 $b_i(\Delta t) = \beta_i - \alpha_i \Delta t$ 的形式。随后，$l$ 直接在小区内搜索出一个满足要求的 $f_i$，并将能够提供 $v m_i$ 的存储节点 $D_i$ 集合下发给 $f_i$。$f_i$ 根据自身的策略选出一个 $d_i \in D_i$，处理完任务后，将大小为 $e_i$ 的计算结果返还给用户 $u_i$。

​	对于一个编号为 $f$ 的计算节点，它的计算能力 $c^f$ 表示其 CPU 频率，按照使用的 CPU 周期数进行计价，单价为 $p_c^f$。其链路延迟 $lt^f$，分配给计算相关文件缓存的上下行链路带宽 $bw^f$，按照使用时间计价，单位价格为 $p_{link}^f$。其可用于缓存 offloaded data 与 VM 的存储空间大小 $S^f$，按照占用的量与时间计价，单位价格为 $p_s^f$。对于一个编号为 $d$ 的存储节点，它会维护一个集合 $VM^d$ 存放它本地持有的虚拟机编号，对外提供出口链路与存储介质两种资源，这些资源有链路延迟 $lt^d$，分配给 VM 上传的链路带宽 $bw^d$，存储介质读取速度 $rd^d$。它们都按照使用时间进行计价，因为二者同时进行，总计单位价格为 $p_{vm}^d$。

​	于是任务完成时间 $\Delta t(i,f^*,d^*)$ 可以被表示为：
$$
\begin{aligned}
\Delta t(i,f^*,d^*) &= t_u(i,f^*) + t_{vm}(i,f^*,d^*) + t_p(i,f^*) + t_d(i,f^*), \\
t_u(i,f^*) &= \frac{s_i}{bw^{uf^*}} + lt^{uf^*}, \\
t_{vm}(i,f^*,d^*) &= \frac{block_1^{vm_i}}{min(bw^{f^*d^*},rd^{d^*})} + lt^{f^*d^*}, \\
t_p(i,f^*) &= \frac{w_i}{c^{f^*}}, \\
t_d(i,f^*) &= \frac{e_i}{bw^{uf^*}} + lt^{uf^*},
\end{aligned}
$$
其中，$f^*$ 是任务 $i$ 所选的计算节点，$d^*$ 是 $f^*$ 选择的存储节点，$t_u(i,f^*)$ 代表任务上传时间，$t_{vm}(i,f^*,d^*)$ 是第一个 VM 区块的加载时间，$block_1^{vm_i}$ 是该区块的大小，$t_p(i,f^*)$ 是任务处理时间，$t_d(i,f^*)$ 是结果下发时间。$bw^{uf^*}$、$lt^{uf^*}$ 是用户与计算节点间链路的带宽与延迟，$bw^{f^*d^*}$、$lt^{f^*d^*}$ 是计算与存储节点间链路的带宽与延迟。注意，对于节点 $i$ 与 $j$，$bw^{ij} = bw^{ji}$，$lt^{ij} = lt^{ji}$。根据以上内容，可以得出计算节点 $f^*$ 处理任务 $i$ 的成本为：
$$
Cost_1^{task}(i,f^*,d^*) = p_c^{f^*} \cdot w_i + p_{link}^{f^*} \cdot [t_{u}(i,f^*) + t_{vm}(i,f^*,d^*) + t_{d}(i,f^*)] \\ 
+ p_s^{f^*} \cdot s_i \cdot [t_{vm}(i,f^*,d^*) + t_p(i,f^*)] + p_s^{f^*} \cdot b\overline{loc}k^{vm_i} \cdot t_p(i,f^*) + p_s^{f^*} \cdot e_i \cdot t_d(i,f^*),
$$
这里的 $b\overline{loc}k^{vm_i}$ 表示 $vm_i$ 平均每个区块的大小，我们假定 $f^*$ 用完一个区块就会立即释放。对于存储节点 $d^*$，由于本模型只涉及 VM 的下载而不涉及对缓存内容的调度，在此我们并不关心它的具体存储开销，因此其完成任务 $i$ 的成本被表示为：
$$
Cost_2^{task}(i,f^*,d^*) = p_{vm}^{d^*} \cdot [t_{vm}(i,f^*,d^*) + t_p(i,f^*)].
$$
任务 $i$ 的总成本 $Cost^{task}(i, f^*, d^*) = Cost_1^{task}(i, f^*, d^*) + Cost_2^{task}(i, f^*, d^*)$。

​	令一个时隙（slot）大小为 $\delta_t$，在时隙 $t$ 下，编号为 $j$ 的节点上，指示变量 $x_{link}(j,t) \in \{0, 1\}$ 表示链路是否被占用（仅当 $j$ 作为计算节点时），$x_{vm}(j,t) \in \{0, 1\}$ 表示是否正在上传 VM，$x_p(j,t) \in [0, \delta_t * c^j]$ 表示该时隙内使用的 CPU 周期数，$x_s(j,t) \in [0, S^j]$ 是该时隙内占用的磁盘空间。于是可以得到该时隙下的总成本：
$$
\begin{aligned}
Cost^{slot}(j,t) &= Cost_1^{slot}(j, t) + Cost_2^{slot}(j, t), \\
Cost_1^{slot}(j, t) &= p_c^j \cdot x_p(j,t) + p_{link}^j \cdot x_{link}(j,t) + p_s^j \cdot x_s(j,t), \\
Cost_2^{slot}(j, t) &= p_{vm}^j \cdot x_{vm}(j,t).
\end{aligned}
$$

### 优化目标与约束条件

​	令给定区域内有 $N_u$ 个用户设备，编号集合为 $U$，它们在给定的时间内发起了 $N_i$ 个任务，参与池化的 $N_m$ 个边缘设备编号集合为 $M$。所有 $N_u+N_m$ 个设备进行独立编号，$U \cap M = \emptyset$。整个问题的**优化目标**建模为在给定 $T$ 个时隙下的 Social Welfare：
$$
max_{x_i,x_{ij}^f,x_{ik}^d} \quad
\sum_{i \in N_i} x_i \cdot b_i( \sum_{j \in N_m} \sum_{k \in N_m} x_{ij}^f x_{ik}^d \cdot \Delta t(i,j,k)) - 

\sum_{j \in N_m} \sum_{t \in T} Cost^{slot}(j, t),
$$

其中，决策变量是三个指示变量：$x_i \in \{0, 1\}$ 指示是否接受任务 $i$、$x_{ij}^f \in \{0, 1\}$ 指示节点 $j$ 是否是任务 $i$ 的计算节点、 $x_{ik}^d \in \{0, 1\}$ 指示节点 $k$ 是否是任务 $i$ 的存储节点。它们的约束条件包括：
$$
\begin{aligned}
S.T. \qquad

\forall i \in [1,N_i],\ \forall k \in N_m,\quad 
&x_{ik}^d \cdot R(sid_i) \in \{0\} \cup VM^k, \\

\forall i \in [1,N_i],\ \forall j \in N_m,\quad  
&x_{ij}^f \cdot [s_i + b\overline{loc}k^{R(sid_i)}] \le S^j,\\

\forall a,b \in [1, N_i],\ \forall j \in N_m,\quad
&x_{aj}^f \cdot t_a \ge x_{bj}^f \cdot [t_b + \sum_{k \in N_m}x_{ik}^d \Delta t(b, j, k)] \ \bigcup \\ &x_{bj}^f \cdot t_b \ge x_{aj}^f \cdot [t_a + \sum_{k \in N_m}x_{ik}^d \Delta t(a, j, k)],\\

\forall i \in [1,N_i],\quad 
&x_i = \sum_{j \in N_m}x_{ij}^f,\ x_i = \sum_{k \in N_m}x_{ik}^d, \\

\forall i \in [1,N_i],\quad 
&0 \le \sum_{j \in N_m}x_{ij}^f \le 1,\ 0 \le \sum_{k \in N_m}x_{ik}^d \le 1, \\

\forall i \in [1,N_i],\ \forall j \in N_m,\quad 
&x_i \in \{0, 1\},\ x_{ij}^f \in \{0, 1\},\ x_{ij}^d \in \{0, 1\}.

\end{aligned}
$$
​	基于这些决策变量，可以得到指示变量的表达式 $x_{link}(j,t)$、$x_{vm}(j,t)$、$x_p(j,t)$、$x_s(j,t)$：

$$
\begin{aligned}

x_{link}(j,t) &= \left\{ 
\begin{array}{lr}
1, & \quad
t_i  \le t < t_i + t_u(i,j) + t_{vm}(i,j)
\ \bigcup \ 
t_i + \Delta t(i,j,k) - t_d(i,j) \le t < t_i + \Delta t(i,j,k)  \\&
|_{\exists i \in [1, N_i],\ \exists k \in N_m,\ x_{ij}^f = 1,\ x_{ik}^d = 1},\\
0, & others.\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad
\end{array}
\right. \\

\\

x_{vm}(k,t) &= \left\{ 
\begin{array}{lr}
1, & \quad \quad \ \ \
t_i + t_u(i,j) \le t < t_i + t_u(i,j) + t_{vm}(i,j,k)  + t_p(i,j) 
|_{\exists i \in [1, N_i],\ \exists j \in N_m,\ x_{ij}^f = 1,\ x_{ik}^d = 1},\\
0, & others.\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad \ \ \ 
\end{array}
\right. \\

\\

x_p(j,t) &= \left\{ 
\begin{array}{lr}
\delta_t \cdot c^j, & 
t_i + t_u(i,j) + t_{vm}(i,j,k) \le\ t < t_i + t_u(i,j) + t_{vm}(i,j,k) \ + \\& t_p(i,j) 
|_{\exists i \in [1, N_i],\ \exists k \in N_m,\ x_{ij}^f = 1,\ x_{ik}^d = 1}, \\
w_i - \delta_t \cdot c^j \cdot [t_p(i,j)-1], &
t_i + \Delta t(i,j,k) - t_d(i,j) -1 \le t < t_i + \Delta t(i,j,k) - t_d(i,j) \\&
|_{\exists i \in [1, N_i],\ \exists k \in N_m,\ x_{ij}^f = 1,\ x_{ik}^d = 1},\\
0, & others.\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\ \ \ 
\end{array}
\right. \\

\\

x_s(j,t) &= \left\{ 
\begin{array}{lr}
s_i, & \ 
t_i + t_u(i,j) \ \le \ t \ < \ t_i +  t_u(i,j) + t_{vm}(i,j,k)
|_{\exists i \in [1, N_i],\ \exists k \in N_m,\ x_{ij}^f = 1,\ x_{ik}^d = 1},\\
s_i + b\overline{loc}k^{vm_i}, &
t_i + t_u(i,j) + t_{vm}(i,j,k) \ \le \ t \ < \ t_i+\ t_u(i,j)\ +\ t_{vm}(i,j,k)\ +\ t_p(i,j) \\&
|_{\exists i \in [1, N_i],\ \exists k \in N_m,\ x_{ij}^f = 1,\ x_{ik}^d = 1},\\
e_i, &
t_i + \Delta t(i,j,k) - t_d(i,j) \le t < t_i + \Delta t(i,j,k)
|_{\exists i \in [1, N_i],\ \exists k \in N_m,\ x_{ij}^f = 1,\ x_{ik}^d = 1},\\
0, \quad \ \ & others.\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad \ \ \ 
\end{array}
\right. \\

\end{aligned}
$$
​	虽然我们能够建模出在 $l$ 上对所有任务进行全局调度的方案，但实际上我们欠缺两个节点 $i$、$j$ 之间的具体的链路信息 $bw^{ij}$、$lt^{ij}$ ，导致不能得到精确的 Social Welfare。换句话说，$l$ 只有 partial observation，因此它的决策变量需要被重新设计。在此过程中，$x_i$ 和 $x_{ij}^f$ 的决策不是本文重点，我们将之固定为一个简单的贪婪算法，然后具体讨论如何在 $l$ 与 $f$ 的 stackelberg 博弈过程中通过决策与 $x_{ik}^d$ 相关的内容来最大化 Social Welfare。

## 基于 Stackelberg 博弈游戏的建模

### Stackelberg 游戏及其均衡性

​	由于欠缺具体的链路信息，$l$ 只能对目标函数值进行估测，比如使用端口信息估计链路信息 $\hat{bw}^{ij} \approx min(bw^i, bw^j)$，$\hat{lt}^{ij} \approx lt^i + lt^j$（对于一个变量 $x$，$\hat{x}$ 在本文中表示对它的估计），然后使用链路估计值求得 Social Welfare。很明显，由于边缘网络的异构性，这种估计目标值与该分配方案的真实目标值之间存在不少误差，尤其是当无线传感器网络（WSN）为两个节点的连接提供其中的一段无线多跳链路时。因此，由 $l$ 根据局部观测信息直接指定 $x_{ij}^f$ 与 $x_{ik}^d$ 的方案在某些复杂网络环境下会带来极其糟糕的用户体验。

​	作为解决方案，$l$ 将只决定 $x_i$ 和 $x_{ij}^f$，之后交给选择出的 $f$ 进行 $x_{ik}^d$ 的决策。这里存在三个假设：

1. 边缘节点在参与其他业务的过程中已经知道与部分其他节点的链路情况。如以前合作的历史信息，或者从属一个运营商的节点自己有维护它们间的链路信息。
2. 边缘节点可能有自己的偏好，但一定是理性的。理性意味着它们会坚持符合自己利益的策略，不会胡乱选择。
3. 边缘节点和 $l$ 的利益不一定是相同的，哪怕 $l$ 谨慎的设置了对它的定价以激励它朝着 Social Welfare 努力。
   - 边缘节点同时还在承接其他计算服务，它额外有一个选择 $\hat{t}_{vm}$ 较大的存储节点的倾向；
   - 边缘节点同时还在承接其他存储服务，它额外有一个选择 $\hat{t}_{vm}$ 较小的存储节点的倾向；
   - 边缘节点来自某个运营商，它会更倾向于选择相同运营商的节点。

​	上述过程很明显是一局 stackelberg 游戏，因为它满足：

1. 存在 leader 与 follower 两个决策主体，各自采用独立的策略，并遵守严格的先后顺序；
2. 二者策略的目标函数都包括对方的决策变量，leader 会在考虑 follower 策略的同时采取最大化自身收益的策略；
3. Follower 的策略是从有限行为集合中选择出单个行为的过程，必定存在让优化目标最大的最佳策略，同时理性的边缘节点会坚持以该最佳策略行动，因此 stackelberg-nash equilibria 存在。

​	按照 stackelberg 的求解思路，$l$ 会在已知 $f$ 最佳策略的情况下求得令自己优化目标最大化的策略。因此，一种直觉的方法是，让 $f$ 的策略成为 $l$ 决策的输入条件之一，$l$ 输出一个影响 $f$ 决策结果的行为。那么，现在存在两个问题：

1. $f$ 的具体行为决策对 $l$ 是未知的，$l$ 如何根据 $f$ 的策略来解决它的最优化问题；
2. $l$ 该通过什么方式来影响 $f$ 的决策内容。

### Leader 建模

​	在一个服务小区内，leader 节点 $l$ 只能获知该区域内某一编号为 $i$ 的节点（包括用户设备与服务提供设备）的自身信息，包括 $c^i$、$S^i$、$bw^i$、$lt^i$、$VM^i$，以及使用各资源的单位价格。同时，$l$ 知晓 $sid$ 与 VM 编号的映射函数 $R(*)$，根据边缘节点编号查找运营商的映射函数  $CSP(*)$，该区域内到达的所有任务信息，某一编号为 $vm_i$ 的 VM 的任意区块 $p$ 的大小 $block_p^{vm_i}$。

​	我们将 leader 的决策设计为在线算法，即每次任务到来时立即处理该任务，而非收集一段时间 $T$ 内的所有任务使用离线算法求出最佳的调度方案。但是，该在线算法对于每一个任务的优化目标依然是围绕着全局 Social Welfare 设计，以避免 $l$ 出现短视的决策。后文将给出基于马尔科夫决策过程的在线算法设计，由于它需要获取每一个任务对总目标的贡献，前述基于 Social Welfare 的**离线算法**的优化目标与约束条件需要被调整为以任务为单位的统计形式：
$$
\begin{aligned}
max_{x_i,x_{ij}^f,x_{ik}^{CA}} \quad
\sum_{i \in N_i} x_i \cdot [ \sum_{j \in N_m} x_{ij}^f \cdot b_i(\Delta &t(i,j,K_{ij})) -
\sum_{j \in N_m} x_{ij}^f \cdot Cost^{task}(i, j, K_{ij})]. \\
\\

S. T. \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad 
K_{ij}& = \mu_f^j(B(i),INFO(x_{ik}^{CA})),\\

INFO(x_{ik}^{CA})=\bigcup_{k \in N_m} &x_{ik}^{CA} \cdot \{ k,\ CSP(k),\ p_{vm}^d,\ min(bw^k,rd^k),\ lt^k \},\\

0 \le & \sum_{j \in N_m} x_{ij}^f \le 1,\ x_i = \sum_{j \in N_m}x_{ij}^f,\\

\forall k \in N_m,\quad &x_{ik}^{CA} \cdot R(sid_i) \in \{0\} \cup VM^k,\\

\forall j \in N_m,\quad  
&x_{ij}^f \cdot [s_i + b\overline{loc}k^{R(sid_i)}] \le S^j,\\

\forall a,b \in [1, N_i],\ \forall j \in N_m,\quad
&x_{aj}^f \cdot t_a \ge x_{bj}^f \cdot [t_b + \Delta t(b, j, K_{ij}] \ \bigcup \\ &x_{bj}^f \cdot t_b \ge x_{aj}^f \cdot [t_a + \Delta t(a, j, K_{ij}],\\

\forall k \in N_m,\quad &x_i \in \{0, 1\},\ x_{ij}^f \in \{0, 1\},\ x_{ik}^{CA} \in \{0, 1\}.
\end{aligned}
$$

其中，决策变量 $x_{ik}^{CA}$ 是 $l$ 为任务 $i$ 选择的备选存储节点的指示变量，$\mu_f^j(*)$ 是节点 $j$ 的 follower 决策函数，它根据任务信息 $B(i)$ 与备选存储节点的信息 $INFO(x_{ik}^{CA})$ 决策出目标存储节点的编号。类似地，$l$ 选择 $x_{ij}^{CA}$ 的决策函数可以被表示为 $\mu_l(*)$，其输入是任务信息 $B(i)$、计算节点的状态信息 $INFO(x_{ij}^f)$、与所有节点的信息 $INFO(\bigcup_{j \in N_m} \{1\})$。值得注意的是，策略 $\mu_f^j(*)$ 内部隐含了节点 $j$ 对某些节点更深层次的信息，譬如它对同一运营商所属的其他节点的详细链路信息，而策略 $\mu_l(*)$ 仅能知晓 $INFO(*)$ 函数中包含的片面节点信息。

​	在这个模型中，$l$ 通过修改 $f$ 能够选择的存储节点来控制其决策，这种方法的本质是在限制 follower 的视野。详细的决策过程包括：

1. $\{x_{ij}^f\}$：在用户就近区域内，贪心地选出资源较为丰富的一个作为 $f$，以提供合适的 QoS；
2. **$\{x_{ik}^{CA}\}$：在所有包含 $vm_i = R(sid_i)$ 的存储节点中选择出一部分备选者（candidates），以控制 $f$ 的决策范围，避免其选择到一些明显偏离 $l$ 目标的节点**；
3. $\{x_i\}$：上述任何一个过程中发生错误，或者根据局部观测信息估计的目标函数值小于 0，都及时终止并抛弃任务。

​	在这个建模中，我们主要讨论 leader 与 follower 节点之间的交互及其对优化问题的影响。因此，我们将重点关注决定决策变量 $\{x_{id}^{CA}\}$ 的策略 $\mu_l(*)$。至于 $\{x_i\}$ 与 $\{x_{ij}^f\}$ 的调度问题，本文固定它们的决策方案如上述流程所示，以排除它们对 stackelberg 游戏的影响。

### 基于试探的 Follower 策略标记方法

​	在游戏中，$f$ 的具体策略将被视为 $l$ 交互环境的一部分，该状态信息以某种形式被 $l$ 的决策函数 $\mu_l(*)$ 所观测。在此，我们提出一种基于试探节点的方案：在每个节点入网时，$l$ 都向它发送一个虚拟任务以及一批虚拟存储节点信息，按照它选择的存储节点编号设置类别信息。其具体过程如下：

1. 生成 $k$ 组随机的测试组合 $ST$，其中的第 $i$ 组表示为 $st_i = \{B_i, INFO_i\}|_{i \in [1,k]} \in ST$，$INFO_i$ 是 $m$ 个虚拟节点的信息，包括的内容如前面同名函数所示；
2. 当节点 $j$ 加入该雾计算系统时，$l$ 依次向其发起上述 $k$ 组测试组合 $ST$；
3. 节点 $j$ 对每个测试样例进行决策，并向 $l$ 返还决策结果（所选的目标存储节点编号）；
4. 每获得一个测试样例的结果，$l$ 都直接向节点 $j$ 发起任务终止的指令，并接着进入下一个测试样例；
5. $l$ 收集这 $k$ 个结果，并将之按顺序排列为长为 $k$ 的向量 $TA_j$。

​	在该方案中，所有存储节点都是虚拟的，因此边缘节点只能用估计的链路信息进行决策，避免异构的网络环境对处在不同区域的同种节点的决策结果的影响。这些虚拟的节点具有随机的运营商所属和上传性能，只要测试数量 $k$ 足够大，就有希望通过决策结果向量对不同的决策类型进行区分。考虑到决策结果同时还会受边缘节点的运行商和接口网络信息影响，于是节点 $j$ 的标签信息可以被表示为它自身信息与测试结果向量的组合 $class_j = \{CSP(j), bw^j, lt^j, TA_j\}$。

​	上述过程所得到的标签信息 $class_j$ 包含了能够对节点 $j$ 策略进行分类的全部信息，但是它并不具有可读性，或者说我们很难通过两个标签信息直接看出它们属于同种策略。一个原因是，采用相同策略的多个节点不可能具有完全一致的条件，它们的测试结果 $TA$ 也不会完全一样。另一个原因则和节点的进行决策的方式有关，若它的策略中包括以某种分布进行的采样，在使用一样的 $ST$ 对其进行多次测试，其生成的 $TA$ 值都可能略有区别。因此，我们采用深度神经网络对节点 $j$ 的标签信息 $class_j$ 进行信息提取。只要训练样本足够，神经网络能够有效解决这类带有误差的分类问题。更具体地，本文采用深度强 化学习拟合 $l$ 节点的决策模型，$class_j$ 将作为决策模型状态观测值的一部分使用。在强化学习进行探索与训练的过程中，$class_j$ 所包含的节点策略信息将被神经网络逐渐理解。

### Follower 建模

​	在一次任务中，$l$ 首先计算决策变量 $\{x_{ij}^f\}$，得到一个目标计算节点 $f$。然后，$l$ 计算出待选存储节点的决策变量 $\{x_{ik}^{CA}\}$，并将与任务和待选节点相关的信息下发给 $f$。$f$ 按照自身的策略，从待选节点中选择出目标存储节点，并将结果告知 $l$。$l$ 向用户通告本次任务的卸载对象 $f$，用户开始计算卸载过程。$f$ 完成任务后，将处理结果递交给用户，并向 $l$ 上报任务完成情况与资源消耗报价（包括自身开销以及存储节点的报价）。用户收到结果后，向 $l$ 核实任务完成情况，并反馈 QoS 信息。

​	对于一个计算节点 $j$，它为任务 $i$ 进行决策的目标函数可以被表示为：
$$
\begin{aligned}
max_{x_k^t} \quad \sum_{k \in N_m}  x_k^t \cdot [Price(i,j,k) &-
 Cost^{task}(i, j, k) + Bias(i,j,k)].\\
\\

S.T. \quad \quad \quad \quad \sum_{k \in N_m} &x_l^t \in \{0, 1\},\\
\forall k \in N_m,&\quad x_k^t \in \{0, 1\}.
\end{aligned}
$$
其中，$Price(i,j,k)$ 表示任务 $i$ 在分配给计算节点 $j$ 与存储节点 $k$ 时，$l$ 将给予节点 $j$ 的收益。$Bias(i,j,k)$ 是节点 $i$ 在处理任务 $i$ 时的偏好函数，不同节点决策类型的差异就反映在该偏好项的设计上。

​	由于本文不讨论定价函数对 $f$ 的影响，同时为了保证 $l$ 与 $f$ 优化目标的关联性，我们采用对用户期望价值的简单缩放来设计定价函数：
$$
Price(i,j,k) = \alpha \cdot b_i(\Delta t(i,j,k)),
$$
其中，$\alpha$ 是缩放系数。值得注意的是。由于 $f$ 在进行决策时不一定知道与存储节点 $k$ 之间的链路情况，该任务的完成情况 $\Delta \hat{t}(i,j,k)$  只能通过 $\hat{bw}^{jk} \approx min(bw^j, bw^k)$ 与 $\hat{lt}^{jk} \approx lt^j + lt^k $ 进行估计，所得定价函数就是基于该估计结果的用户期望价值的缩放。

​	对于偏好函数，本文设计四种策略：

1. 无偏好：$Bias_0(i,j,k) = 0$；
2. 计算保守型：边缘节点同时还在承接其他计算服务，它希望降低单位时间内的计算资源占用，$Bias_1(i,j,k) = - \beta \cdot \frac{t_c}{\Delta \hat{t}(i,j,k)}$；
3. 存储保守型：边缘节点同时还在承接其他存储服务，它希望减少存储资源的占用时间，$Bias_2(i,j,k) = - \beta \cdot \frac{t_u + \hat{t}_{vm}+t_d}{\Delta \hat{t}(i,j,k)}$；
4. 同运营商保护型：边缘节点倾向于选择相同运营商的节点，$Bias_3(i,j,k) = \left\{ \begin{aligned} &\beta \cdot \Mu, \quad j=k,\\ &0, \quad \quad \ others. \end{aligned} \right.$

​	上述表达式中，$\beta$ 是偏好项的系数，$\Mu$ 是一个较大的正实数。实际上，考虑到 $f$ 对存储节点的选择只会影响 VM 的下载时间，计算和存储保守型策略的偏好函数又可以分别表示为 $Bias_1(i,j,k) = \beta \cdot \hat{t}_{vm}$ 和 $Bias_2(i,j,k) = - \beta \cdot \hat{t}_{vm}$。

## 基于马尔科夫决策过程的决策模型

### Follower 建模

​	由于作为 $f$ 的节点 $j$ 的策略本质上是根据时隙 $t$ 中接收到的任务 $i$ 与备选节点的信息 $obs_t^f(i) = \{B(i),INFO_{ij}^{CA}\}$ 选出一个目标节点 $K_{ij} = \mu_f^j(obs_f)$，我们可以将它的决策过程建模成马尔科夫决策过程（MDP）。它的环境观测值就是 $obs_t^f(i)$，状态空间的维度根据待选节点的数量 $m_i^{CA}$ 变化。行为是  $m_i^{CA}$ 个在 $[0,\ 1]$ 取值的实数组成的独热码（one-hot code），其中值最大的数所对应的节点作为选择出的目标存储节点。为了最大化它自己的优化目标，其回报函数为 $R_t^f(i,j,k) = Price(i,j,k) - Cost^{task}(i, j, k) + Bias(i,j,k)$。由于边缘节点的优化目标为 $G_t^f=R_t^f$，意味着 $G_t^f=R_t^f + \sum_{i =1}^{+\infty} 0^i \cdot R_{t+i}^j$，即该过程是一个单步马尔科夫决策过程。

​	在本文中，虽然我们并不关注 $l$ 的具体实现，只假设它使用的深度强化学习（DRL）算法进行决策，但是为了证明该建模的可实践性，这里给出这种状态空间与行为空间均不稳定的 MDP 模型解决方案：使用 Attention 网络作为 DRL 智能体的决策模型，在第一层输入任务信息，其后的不定长层依次输入 $m_i^{CA}$ 个节点的信息，然后吐出 $m_i^{CA}$ 个 $[0,\ 1]$ 取值的数作为独热码。

### Leader 建模

​	通过将 $f$ 的行为建模为 MDP，自然而然就得到一种 Stackelberg 游戏的交互思路：由 $l$ 控制 $f$ 的环境观测值 $obs^f$，进而让 $f$ 向着对 $l$ 有利的方向行动。在我们的设计中，$l$ 所控制的就是 $f$ 能够观测到的备选存储节点的状态值，因此本文的建模本质上就是一个智能体 $l$ 通过控制另一个智能体 $f$ 的局部环境观测值、进而间接控制它的决策的双智能体马尔可夫决策过程。在该问题中，编号为 $j$ 的 $f$ 的具体决策对 $l$ 是不可知的，它只能通过基于试探得到的 $class_j$ 去估计策略 $\mu_f^j$。

​	不同于 $f$，$l$ 的优化目标是全局的 Social Welfare，它的决策需要考虑多步以后的价值函数。因此，其回报函数为当前任务 $i$ 对 Social Welfare 的贡献 $R_t^l = b_i(\Delta t(i,f^*,K_{if^*})) - Cost^{task}(i, j, K_{if^*})$，**在线算法**在时隙 $t$ 的优化目标为：
$$
G_t^l=R_t^l + \sum_{i =1}^{+\infty} \gamma^i \cdot R_{t+i}^l,
$$
其中，$f^*$ 是 $l$ 选出的计算节点，视作环境的一部分，$\gamma$ 为折扣因子（discounted factor）。为了简化求解，$l$ 的状态空间与行为空间需要是固定的。因此，我们将行为设计为对 $N_m$ 个节点的选择概率分布的参数，节点 $i$ 的选择概率由一个均值 $\mu_i$ 和方差 $\sigma_i$ 的高斯分布采样后，再经过 Sigmoid 函数归一化后得出。$l$ 的行为空间就是这 $2N_m$ 个参数，从头到尾依次表示每个节点采样使用的均值和方差。同时，状态观测值 $obs_t^l(i) = \{B(i),INFO_{if^*}^{f},INFO(\bigcup_{j \in N_m} \{1\})\}$ 也是定长的，它的状态空间是稳定的。

​	作为一个 Stackelberg 游戏中的 leader，按照传统的求解思路，$l$ 的决策就是在给定 $f$ 的最优策略的情况下，通过求解约束优化问题得到自己的最佳策略。因此，它的最佳策略可以被表示为：
$$
\mu_l^*(obs_t^l)\ =\ max_{a_t}\ E\ [\ G_t^l\ |\ obs_t^l,\ a_t,\ \hat\mu_f^j\ ],
$$
由于 $l$ 无法知晓 $f$ 的真正策略，它只能使用 $\hat\mu_f^j \approx class_j$ 进行估计。如果将 $obs_t^l(i)$ 与 $\hat\mu_f^j$ 进行合并，得到 $obs_t^{l'} = \{B(i),INFO_{if^*}^{f}, TA_{f^*},INFO(\bigcup_{j \in N_m} \{1\})\}$，那么有：
$$
\mu_l^*(obs_t^l)\ =\ max_{a_t}\ E\ [\ G_t^l\ |\ obs_t^{l'},\ a_t\ ],
$$
很明显，该式正是强化学习算法求解 MDP 时的追求目标策略，意味着我们可以直接使用以 $obs_t^{l'}$ 为观测值的深度强化学习算法去求解该 Stackelberg 游戏中的 leader 策略。

## Experiment

### 实验设计

- 任务丢弃
  - 采用应收尽收的原则，不主动丢弃任务
  - 只有在资源不满足时才舍弃任务
  - 因为本文的核心是 $l$ 与 $f$ 的交互，所以只要固定这部分策略，就不影响我们的实验

- 独占性
  - $f$ 被分配给某个任务后，直到任务结束为止，就不能再分配给其他任务
  - $d$ 不应该被独占，否则需要有比用户数量多的 $d$
- 选择 $f$ 时只考虑没有接受任务的节点，并判断其存储空间是否合适
- 选择 $d$ 时不管它是否正在繁忙，一律认为可用
  - 实际上 $d$ 只消耗硬盘读取与网络上传资源，我们假设所有节点的总能力远大于分配给一个服务的 bw 和 rd 资源，这个设计就是可解释的
  - 这种简化，避免了我们在某个 slot 里的决策顺序问题，必定会有不错的训练效果（在论文中是否需要提及？）
- $l$ 的行为
  - 由于 $N_m$ 很大，实际上 $l$ 并不会从中直接选择 candidates
  - 我们先进行一次预处理，筛选出符合要求的所有节点
  - 再将从这些节点按照性价比（单价/min(bw, rd)）选出靠前的 n_ca 个节点，若未满则补空（价格 1e6，速度 0），按序排列
  - $l$ 的决策函数输出的是这 n_ca 个节点各自的概率分布，将它们分别手动高斯采样（为了有效反向传播，先采样出标准正态分布 $N_0$，再 $\mu + N_0*\sigma$ 得到目标），采样结果 sigmoid 归一化后超过 0.5 就是设为 candidate
- 与论文不同的细节
  - 仿真中不区分 sid 和 vm，二者一样
  - 仿真中不区分节点的 bw 和 lt，认为一个节点对于 VM 上传和计算卸载分配的资源一样，同时由于 VM 上传不会被视为对带宽资源的占用，这种设计在理论上能够解释

- 节点自己就有 vm 的情况
  - 不执行调度，计价不包括 block 缓存与存储节点的部分


### 问题

1. 环境太简单了，没有对资源使用量上限的设计
   1. 一个服务器同时只能服务一个用户，不太合理
      1. 只要将 slot 切得很小，同一时刻就不会有太多用户请求
      2. 任务大小等参数设置应该围绕 slot，让大部分任务在 2-3 个 slots 内解决
   2. 先做一个简单的跑着看看

### 实验参数

- 我们假设平均单个数据块大小为 10MB 左右，计算结果大小为 1KB。