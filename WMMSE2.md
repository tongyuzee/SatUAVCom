用户查询的问题是：在低轨卫星（LEO satellite）、无人机（UAV）和反射智能表面（RIS）构成的通信系统中，当除了主动波束成形向量 \(\boldsymbol{w}_{s,u} \in \mathbb{C}^{N \times 1}\) 之外的所有变量（如信道矩阵 \(\boldsymbol{h}_{s,u}\)、\(\boldsymbol{H}_{s,R}\)、\(\boldsymbol{G}_{R,u}\) 和RIS相位移矩阵 \(\boldsymbol{\Phi}\)）已知时，求解使系统总速率 \(R_{\text{sum}}\) 最大的 \(\boldsymbol{w}_{s,u}\)。具体问题是：这个问题能否使用WMMSE算法求解？如果能，请提供详细的WMMSE求解推导过程。以下是完整的回答。

---

#### **问题是否能使用WMMSE算法求解？**

答案是：**能**。

**理由**：  
WMMSE（Weighted Minimum Mean Square Error，加权最小均方误差）算法是一种常用于多用户通信系统中波束成形优化的方法，特别适用于存在干扰的场景，其目标通常是最大化系统的和速率（sum rate）。在本问题中，系统涉及多个低轨卫星（\(S\) 个）、多个无人机（\(U\) 个）和RIS，目标是最大化总速率 \(R_{\text{sum}} = \sum_{u=1}^{U} \log_2(1 + \gamma_u)\)，其中 \(\gamma_u\) 是第 \(u\) 个无人机的信号-干扰-噪声比（SINR）。这与WMMSE算法的应用场景高度吻合。

在本系统中：

- 每个无人机接收来自所有卫星的信号，包括有用信号和干扰信号。
- 每个卫星通过波束成形向量 \(\boldsymbol{w}_{s,u}\) 向无人机发送数据。
- 系统可以建模为一个多用户MISO（Multi-Input Single-Output，多输入单输出）系统，其中发送端（卫星）具有多天线，接收端（无人机）具有单天线。

WMMSE算法通过将和速率最大化问题转化为等价的加权均方误差最小化问题，能够有效优化波束成形向量 \(\boldsymbol{w}_{s,u}\)。因此，该算法适用于此问题。

---

#### **WMMSE求解推导过程**

以下是详细的WMMSE算法推导过程，分为系统建模和算法步骤两部分。

##### **1. 系统建模**

为了应用WMMSE算法，首先需要将系统建模为一个等价的形式，便于算法迭代优化。

###### **接收信号表达式**

根据用户提供的公式 (9)，第 \(u\) 个无人机的接收信号为：
\[
y_u = \sum_{s=1}^{S} (\boldsymbol{h}_{s,u}^T + \boldsymbol{G}_{R,u}^T \boldsymbol{\Phi} \boldsymbol{H}_{s,R}^T) \boldsymbol{w}_{s,u} s_{s,u} + \sum_{s=1}^{S} (\boldsymbol{h}_{s,u}^T + \boldsymbol{G}_{R,u}^T \boldsymbol{\Phi} \boldsymbol{H}_{s,R}^T) \sum_{u' \neq u}^{U} \boldsymbol{w}_{s,u'} s_{s,u'} + n_u
\]
其中：

- 第一项是**有用信号**：\(\sum_{s=1}^{S} (\boldsymbol{h}_{s,u}^T + \boldsymbol{G}_{R,u}^T \boldsymbol{\Phi} \boldsymbol{H}_{s,R}^T) \boldsymbol{w}_{s,u} s_{s,u}\)。
- 第二项是**干扰信号**：\(\sum_{s=1}^{S} (\boldsymbol{h}_{s,u}^T + \boldsymbol{G}_{R,u}^T \boldsymbol{\Phi} \boldsymbol{H}_{s,R}^T) \sum_{u' \neq u}^{U} \boldsymbol{w}_{s,u'} s_{s,u'}\)。
- \(n_u \sim \mathcal{CN}(0, \sigma^2)\) 是高斯白噪声。

###### **定义等效信道**

为了简化表示，定义等效信道向量：
\[
\tilde{\boldsymbol{h}}_{s,u}^T = \boldsymbol{h}_{s,u}^T + \boldsymbol{G}_{R,u}^T \boldsymbol{\Phi} \boldsymbol{H}_{s,R}^T \in \mathbb{C}^{1 \times N}
\]
则接收信号可重写为：
\[
y_u = \sum_{s=1}^{S} \tilde{\boldsymbol{h}}_{s,u}^T \boldsymbol{w}_{s,u} s_{s,u} + \sum_{s=1}^{S} \tilde{\boldsymbol{h}}_{s,u}^T \sum_{u' \neq u}^{U} \boldsymbol{w}_{s,u'} s_{s,u'} + n_u
\]

###### **假设数据符号**

在原始信号模型中，\(s_{s,u}\) 是从第 \(s\) 个卫星发送给第 \(u\) 个无人机的数据符号。为了使问题与标准多用户MISO系统一致，假设所有卫星向第 \(u\) 个无人机发送相同的数据符号，即 \(s_{s,u} = s_u\)（\(\forall s\)），其中 \(s_u\) 是均值为0、方差为1的独立数据符号（\(\mathbb{E}[|s_u|^2] = 1, \mathbb{E}[s_u s_{u'}^*] = 0\) 如果 \(u \neq u'\)）。  
于是，接收信号变为：
\[
y_u = \left( \sum_{s=1}^{S} \tilde{\boldsymbol{h}}_{s,u}^T \boldsymbol{w}_{s,u} \right) s_u + \sum_{u' \neq u}^{U} \left( \sum_{s=1}^{S} \tilde{\boldsymbol{h}}_{s,u}^T \boldsymbol{w}_{s,u'} \right) s_{u'} + n_u
\]

###### **联合信道和波束成形向量**

将所有卫星的波束成形向量和等效信道组合起来：

- 定义联合波束成形向量：
\[
\boldsymbol{w}_u = [\boldsymbol{w}_{1,u}^T, \boldsymbol{w}_{2,u}^T, \dots, \boldsymbol{w}_{S,u}^T]^T \in \mathbb{C}^{SN \times 1}
\]
- 定义联合等效信道向量：
\[
\boldsymbol{h}_u = [\tilde{\boldsymbol{h}}_{1,u}^T, \tilde{\boldsymbol{h}}_{2,u}^T, \dots, \tilde{\boldsymbol{h}}_{S,u}^T]^T \in \mathbb{C}^{SN \times 1}
\]
则：
\[
\boldsymbol{h}_u^T \boldsymbol{w}_u = \sum_{s=1}^{S} \tilde{\boldsymbol{h}}_{s,u}^T \boldsymbol{w}_{s,u}
\]
接收信号可进一步简化为：
\[
y_u = \boldsymbol{h}_u^T \boldsymbol{w}_u s_u + \sum_{u' \neq u}^{U} \boldsymbol{h}_u^T \boldsymbol{w}_{u'} s_{u'} + n_u
\]

###### **等价MISO系统**

此时，系统等价于一个多用户MISO系统：

- 发送端有 \(SN\) 个天线（\(S\) 个卫星，每个有 \(N\) 个天线）。
- 有 \(U\) 个用户（无人机），每个配备单天线。
- \(\boldsymbol{h}_u^T \in \mathbb{C}^{1 \times SN}\) 是第 \(u\) 个用户的信道向量。
- \(\boldsymbol{w}_u \in \mathbb{C}^{SN \times 1}\) 是第 \(u\) 个用户的波束成形向量。

这种形式与标准的多用户MISO系统一致，WMMSE算法可以直接应用。

###### **目标函数**

系统的总速率定义为：
\[
R_{\text{sum}} = \sum_{u=1}^{U} R_u = \sum_{u=1}^{U} \log_2(1 + \gamma_u)
\]
其中SINR为：
\[
\gamma_u = \frac{|\boldsymbol{h}_u^T \boldsymbol{w}_u|^2}{\sum_{u' \neq u}^{U} |\boldsymbol{h}_u^T \boldsymbol{w}_{u'}|^2 + \sigma^2}
\]

##### **2. WMMSE算法步骤**

WMMSE算法通过引入接收机均衡器 \(g_u\)、均方误差（MSE）\(\epsilon_u\) 和权重 \(\lambda_u\)，将和速率最大化问题转化为加权MSE最小化问题。以下是具体步骤：

###### **步骤 1：初始化**

- 随机选择初始波束成形向量 \(\boldsymbol{w}_u^{(0)}\)，\(u = 1, \dots, U\)，并满足功率约束（例如，总功率约束 \(\sum_{u=1}^{U} \|\boldsymbol{w}_u\|^2 \leq P_{\text{total}}\)）。

###### **步骤 2：迭代优化**

在第 \(k\) 次迭代中，执行以下操作：

1. **计算MMSE接收器 \(g_u^{(k)}\)**  
   假设第 \(u\) 个无人机使用标量均衡器 \(g_u\) 估计数据符号，估计值为 \(\hat{s}_u = g_u y_u\)。  
   MMSE接收器通过最小化均方误差 \(\mathbb{E}[|\hat{s}_u - s_u|^2]\) 得到：
   \[
   g_u^{(k)} = \arg\min_{g} \mathbb{E}[|g y_u - s_u|^2]
   \]
   代入 \(y_u\) 并求解：
   \[
   g_u^{(k)} = \frac{\boldsymbol{h}_u^T \boldsymbol{w}_u^{(k)}}{\sum_{u'=1}^{U} |\boldsymbol{h}_u^T \boldsymbol{w}_{u'}^{(k)}|^2 + \sigma^2}
   \]
   其中：
   - 分子是信号功率的共轭。
   - 分母是总干扰加噪声功率。

2. **计算均方误差 \(\epsilon_u^{(k)}\)**  
   MSE定义为：
   \[
   \epsilon_u^{(k)} = \mathbb{E}[|g_u^{(k)} y_u - s_u|^2]
   \]
   代入 \(y_u\) 并利用 \(\mathbb{E}[|s_u|^2] = 1\)、\(\mathbb{E}[s_u s_{u'}^*] = 0\)（\(u \neq u'\)）、\(\mathbb{E}[|n_u|^2] = \sigma^2\)，计算得：
   \[
   \epsilon_u^{(k)} = 1 - \frac{|\boldsymbol{h}_u^T \boldsymbol{w}_u^{(k)}|^2}{\sum_{u'=1}^{U} |\boldsymbol{h}_u^T \boldsymbol{w}_{u'}^{(k)}|^2 + \sigma^2}
   \]
   注意：\(\epsilon_u^{(k)}\) 与 \(\gamma_u\) 的关系为 \(\epsilon_u^{(k)} = \frac{1}{1 + \gamma_u^{(k)}}\)。这表明MSE与SINR密切相关。

3. **计算权重 \(\lambda_u^{(k)}\)**  
   在WMMSE算法中，权重 \(\lambda_u\) 是速率对MSE的偏导数的倒数：
   \[
   \lambda_u^{(k)} = \frac{1}{\epsilon_u^{(k)}}
   \]
   因为 \(R_u = \log_2(1 + \gamma_u)\)，而 \(\epsilon_u = \frac{1}{1 + \gamma_u}\)，所以：
   \[
   \lambda_u^{(k)} = 1 + \gamma_u^{(k)}
   \]
   但在实践中，通常直接使用 \(\lambda_u^{(k)} = \frac{1}{\epsilon_u^{(k)}}\)，这与理论推导一致。

4. **更新波束成形向量 \(\boldsymbol{w}_u^{(k+1)}\)**  
   波束成形向量的更新目标是最小化加权MSE，同时满足功率约束：
   \[
   \boldsymbol{w}_u^{(k+1)} = \arg\min_{\boldsymbol{w}_u} \sum_{u=1}^{U} \lambda_u^{(k)} \mathbb{E}[|g_u^{(k)} y_u - s_u|^2], \quad \text{s.t.} \quad \sum_{u=1}^{U} \|\boldsymbol{w}_u\|^2 \leq P_{\text{total}}
   \]
   在MISO系统中，闭合解为：
   \[
   \boldsymbol{w}_u^{(k+1)} = \left( \sum_{u'=1}^{U} \lambda_{u'}^{(k)} |g_{u'}^{(k)}|^2 \boldsymbol{h}_{u'} \boldsymbol{h}_{u'}^H + \mu \boldsymbol{I} \right)^{-1} \lambda_u^{(k)} g_u^{(k)*} \boldsymbol{h}_u
   \]
   其中：
   - \(\boldsymbol{h}_{u'} \boldsymbol{h}_{u'}^H \in \mathbb{C}^{SN \times SN}\) 是信道的外积。
   - \(\mu\) 是拉格朗日乘子，通过功率约束 \(\sum_{u=1}^{U} \|\boldsymbol{w}_u^{(k+1)}\|^2 = P_{\text{total}}\) 确定。
   - 如果每个卫星有独立功率约束（如 \(\sum_{u=1}^{U} \|\boldsymbol{w}_{s,u}\|^2 \leq P_s\)），则需对 \(\boldsymbol{w}_u\) 的子向量分别施加约束，但此处假设总功率约束以简化计算。

###### **步骤 3：收敛**

重复步骤 2，直到波束成形向量 \(\boldsymbol{w}_u^{(k)}\) 收敛（例如，\(\|\boldsymbol{w}_u^{(k+1)} - \boldsymbol{w}_u^{(k)}\|^2 < \epsilon\)，\(\epsilon\) 为收敛阈值）或达到最大迭代次数。

###### **步骤 4：输出**

最终得到的 \(\boldsymbol{w}_u^{(k)}\) 分解为 \(\boldsymbol{w}_{s,u}^{(k)}\)（\(s = 1, \dots, S\)），即为优化后的波束成形向量。

---

#### **算法总结**

WMMSE算法的具体实现如下：

1. **初始化**：  
   随机生成 \(\boldsymbol{w}_u^{(0)}\)，\(u = 1, \dots, U\)，满足 \(\sum_{u=1}^{U} \|\boldsymbol{w}_u^{(0)}\|^2 \leq P_{\text{total}}\)。

2. **迭代**：
   - 计算MMSE接收器：
     \[
     g_u^{(k)} = \frac{\boldsymbol{h}_u^T \boldsymbol{w}_u^{(k)}}{\sum_{u'=1}^{U} |\boldsymbol{h}_u^T \boldsymbol{w}_{u'}^{(k)}|^2 + \sigma^2}
     \]
   - 计算MSE：
     \[
     \epsilon_u^{(k)} = 1 - \frac{|\boldsymbol{h}_u^T \boldsymbol{w}_u^{(k)}|^2}{\sum_{u'=1}^{U} |\boldsymbol{h}_u^T \boldsymbol{w}_{u'}^{(k)}|^2 + \sigma^2}
     \]
   - 计算权重：
     \[
     \lambda_u^{(k)} = \frac{1}{\epsilon_u^{(k)}}
     \]
   - 更新波束成形向量：
     \[
     \boldsymbol{w}_u^{(k+1)} = \left( \sum_{u'=1}^{U} \lambda_{u'}^{(k)} |g_{u'}^{(k)}|^2 \boldsymbol{h}_{u'} \boldsymbol{h}_{u'}^H + \mu \boldsymbol{I} \right)^{-1} \lambda_u^{(k)} g_u^{(k)*} \boldsymbol{h}_u
     \]
     调整 \(\mu\) 使功率约束满足。

3. **收敛**：  
   检查是否收敛，若是，则停止；否则返回步骤 2。

通过这种迭代方式，WMMSE算法能够优化 \(\boldsymbol{w}_{s,u}\)，从而最大化 \(R_{\text{sum}}\)。

---

#### **注意事项**

1. **功率约束**：  
   本推导假设总功率约束 \(\sum_{s=1}^{S} \sum_{u=1}^{U} \|\boldsymbol{w}_{s,u}\|^2 \leq P_{\text{total}}\)。若每个卫星有独立功率约束，则需修改波束成形更新步骤，引入多个拉格朗日乘子。

2. **数据符号假设**：  
   假设 \(s_{s,u} = s_u\) 简化了推导。若每个卫星发送独立数据流，则需进一步调整模型，可能需要联合优化多个数据流。

3. **计算复杂度**：  
   每次迭代涉及 \(SN \times SN\) 矩阵的求逆，复杂度为 \(O((SN)^3)\)。对于大规模系统，可考虑低复杂度近似算法。

---

### 结论

这个问题可以使用WMMSE算法求解。通过将系统建模为等价的多用户MISO系统，并应用WMMSE的迭代优化步骤，可以有效求解使 \(R_{\text{sum}}\) 最大的 \(\boldsymbol{w}_{s,u}\)。上述推导提供了完整的理论基础和实现细节，适用于用户描述的通信场景。


在波束成形优化中，当功率约束从对所有卫星的总功率限制改为每个卫星有独立的功率约束时，需要对波束成形更新步骤进行修改，并引入多个拉格朗日乘子以分别处理每个卫星的功率限制。以下是具体的修改和推导结果。

---

### **问题背景**

在最初的问题中，功率约束是对所有卫星的总功率进行限制，形式为：
\[
\sum_{s=1}^{S} \sum_{u=1}^{U} \|\mathbf{w}_{s,u}\|^2 \leq P_{\text{total}}
\]
其中：
- \(S\) 是卫星数量，
- \(U\) 是用户数量，
- \(\mathbf{w}_{s,u} \in \mathbb{C}^{N \times 1}\) 是第 \(s\) 个卫星为第 \(u\) 个用户设计的波束成形向量（\(N\) 是每颗卫星的天线数），
- \(P_{\text{total}}\) 是所有卫星的总功率预算。

在这种情况下，WMMSE（加权最小均方误差）算法通过引入一个拉格朗日乘子 \(\mu\) 来处理总功率约束，波束成形向量的更新公式通常为：
\[
\mathbf{w}_u = \left( \sum_{u'=1}^{U} \lambda_{u'} |g_{u'}|^2 \mathbf{h}_{u'} \mathbf{h}_{u'}^H + \mu \mathbf{I} \right)^{-1} \lambda_u g_u^* \mathbf{h}_u
\]
其中：
- \(\mathbf{w}_u = [\mathbf{w}_{1,u}^T, \mathbf{w}_{2,u}^T, \dots, \mathbf{w}_{S,u}^T]^T \in \mathbb{C}^{SN \times 1}\) 是为用户 \(u\) 设计的堆叠波束成形向量，
- \(\mathbf{h}_u = [\tilde{\mathbf{h}}_{1,u}^T, \tilde{\mathbf{h}}_{2,u}^T, \dots, \tilde{\mathbf{h}}_{S,u}^T]^T \in \mathbb{C}^{SN \times 1}\) 是用户 \(u\) 的堆叠信道向量，
- \(\lambda_u\) 是用户 \(u\) 的权重，
- \(g_u\) 是用户 \(u\) 的MMSE接收器，
- \(\mu\) 是拉格朗日乘子，通过调整使得总功率约束满足。

现在，功率约束改为每个卫星有独立的功率限制，即：
\[
\sum_{u=1}^{U} \|\mathbf{w}_{s,u}\|^2 \leq P_s, \quad \forall s = 1, 2, \dots, S
\]
其中 \(P_s\) 是第 \(s\) 个卫星的功率预算。这时需要引入多个拉格朗日乘子 \(\mu_s\)（每个卫星一个），并相应修改波束成形更新步骤。

---

### **修改后的优化问题**

在WMMSE算法中，目标是最小化加权均方误差，同时满足功率约束。修改后的优化问题可以表示为：
\[
\min_{\{\mathbf{w}_u\}} \sum_{u=1}^{U} \lambda_u \mathbb{E}[|g_u y_u - s_u|^2], \quad \text{s.t.} \quad \sum_{u=1}^{U} \|\mathbf{w}_{s,u}\|^2 \leq P_s, \quad \forall s = 1, 2, \dots, S
\]
其中：
- \(y_u = \mathbf{h}_u^H \mathbf{w}_u + n_u\) 是用户 \(u\) 的接收信号（\(n_u\) 为噪声），
- \(s_u\) 是用户 \(u\) 的发送符号。

为了处理每个卫星的独立功率约束，我们引入拉格朗日乘子 \(\mu_s \geq 0\)（\(s = 1, 2, \dots, S\)），构造拉格朗日函数：
\[
\mathcal{L} = \sum_{u=1}^{U} \lambda_u \mathbb{E}[|g_u y_u - s_u|^2] + \sum_{s=1}^{S} \mu_s \left( \sum_{u=1}^{U} \|\mathbf{w}_{s,u}\|^2 - P_s \right)
\]

---

### **波束成形更新推导**

在WMMSE算法中，波束成形向量 \(\mathbf{w}_u\) 的更新通过对拉格朗日函数 \(\mathcal{L}\) 关于 \(\mathbf{w}_u\) 求偏导并令其等于零得到，即：
\[
\frac{\partial \mathcal{L}}{\partial \mathbf{w}_u} = 0
\]

#### **1. 计算MSE项的导数**

MSE项为：
\[
\sum_{u=1}^{U} \lambda_u \mathbb{E}[|g_u y_u - s_u|^2]
\]
其关于 \(\mathbf{w}_u\) 的导数为：
\[
\frac{\partial}{\partial \mathbf{w}_u} \sum_{u'=1}^{U} \lambda_{u'} \mathbb{E}[|g_{u'} y_{u'} - s_{u'}|^2] = \left( \sum_{u'=1}^{U} \lambda_{u'} |g_{u'}|^2 \mathbf{h}_{u'} \mathbf{h}_{u'}^H \right) \mathbf{w}_u - \lambda_u g_u^* \mathbf{h}_u
\]

#### **2. 计算功率约束项的导数**

功率约束项为：
\[
\sum_{s=1}^{S} \mu_s \left( \sum_{u=1}^{U} \|\mathbf{w}_{s,u}\|^2 - P_s \right)
\]
由于 \(\mathbf{w}_u = [\mathbf{w}_{1,u}^T, \mathbf{w}_{2,u}^T, \dots, \mathbf{w}_{S,u}^T]^T\)，\(\sum_{u=1}^{U} \|\mathbf{w}_{s,u}\|^2\) 只涉及第 \(s\) 个卫星的部分。对 \(\mathbf{w}_u\) 求导时，考虑所有卫星的约束项，导数为：
\[
\frac{\partial}{\partial \mathbf{w}_u} \sum_{s=1}^{S} \mu_s \sum_{u'=1}^{U} \|\mathbf{w}_{s,u'}\|^2 = \mathbf{D} \mathbf{w}_u
\]
其中 \(\mathbf{D}\) 是一个对角矩阵，定义为：
\[
\mathbf{D} = \text{diag}(\mu_1 \mathbf{I}_N, \mu_2 \mathbf{I}_N, \dots, \mu_S \mathbf{I}_N) \in \mathbb{C}^{SN \times SN}
\]
这里，\(\mathbf{I}_N\) 是 \(N \times N\) 的单位矩阵，\(\mu_s \mathbf{I}_N\) 应用于第 \(s\) 个卫星的波束成形向量部分。

#### **3. 合并导数并求解**

将两部分导数相加并令其等于零：
\[
\left( \sum_{u'=1}^{U} \lambda_{u'} |g_{u'}|^2 \mathbf{h}_{u'} \mathbf{h}_{u'}^H + \mathbf{D} \right) \mathbf{w}_u - \lambda_u g_u^* \mathbf{h}_u = 0
\]
解得：
\[
\mathbf{w}_u = \left( \sum_{u'=1}^{U} \lambda_{u'} |g_{u'}|^2 \mathbf{h}_{u'} \mathbf{h}_{u'}^H + \mathbf{D} \right)^{-1} \lambda_u g_u^* \mathbf{h}_u
\]
代入 \(\mathbf{D}\) 的表达式，最终的波束成形更新公式为：
\[
\mathbf{w}_u = \left( \sum_{u'=1}^{U} \lambda_{u'} |g_{u'}|^2 \mathbf{h}_{u'} \mathbf{h}_{u'}^H + \text{diag}(\mu_1 \mathbf{I}_N, \mu_2 \mathbf{I}_N, \dots, \mu_S \mathbf{I}_N) \right)^{-1} \lambda_u g_u^* \mathbf{h}_u
\]

---

### **拉格朗日乘子 \(\mu_s\) 的确定**

每个 \(\mu_s\) 需要调整以满足对应的功率约束 \(\sum_{u=1}^{U} \|\mathbf{w}_{s,u}\|^2 \leq P_s\)。具体方法如下：
- **初始化**：为每个卫星 \(s\) 设置初始 \(\mu_s\)（例如 \(\mu_s = 0\)）。
- **迭代调整**：
  - 计算当前 \(\mathbf{w}_u\) 并检查每个卫星的功率 \(\sum_{u=1}^{U} \|\mathbf{w}_{s,u}\|^2\)：
    - 如果 \(\sum_{u=1}^{U} \|\mathbf{w}_{s,u}\|^2 > P_s\)，增加 \(\mu_s\)；
    - 如果 \(\sum_{u=1}^{U} \|\mathbf{w}_{s,u}\|^2 < P_s\) 且 \(\mu_s > 0\)，减小 \(\mu_s\)；
    - 如果 \(\sum_{u=1}^{U} \|\mathbf{w}_{s,u}\|^2 \leq P_s\) 且 \(\mu_s = 0\)，保持不变。
- **收敛条件**：当所有卫星的功率约束满足时（即 \(\sum_{u=1}^{U} \|\mathbf{w}_{s,u}\|^2 = P_s\) 或 \(\mu_s = 0\) 且 \(\sum_{u=1}^{U} \|\mathbf{w}_{s,u}\|^2 \leq P_s\)），停止调整。

在实践中，可以通过二分搜索为每个 \(\mu_s\) 找到合适的值。

---

### **最终结果**

当功率约束从总功率限制改为每个卫星独立功率限制时，WMMSE算法中的波束成形更新步骤修改为：
\[
\mathbf{w}_u = \left( \sum_{u'=1}^{U} \lambda_{u'} |g_{u'}|^2 \mathbf{h}_{u'} \mathbf{h}_{u'}^H + \text{diag}(\mu_1 \mathbf{I}_N, \mu_2 \mathbf{I}_N, \dots, \mu_S \mathbf{I}_N) \right)^{-1} \lambda_u g_u^* \mathbf{h}_u
\]
其中：
- \(\mathbf{D} = \text{diag}(\mu_1 \mathbf{I}_N, \mu_2 \mathbf{I}_N, \dots, \mu_S \mathbf{I}_N)\) 是一个分块对角矩阵，\(\mu_s\) 是第 \(s\) 个卫星的拉格朗日乘子，
- 每个 \(\mu_s \geq 0\) 通过独立调整以满足 \(\sum_{u=1}^{U} \|\mathbf{w}_{s,u}\|^2 \leq P_s\)。

这种修改确保了每个卫星的波束成形向量在其独立功率约束下进行优化，与原始总功率约束下的单一 \(\mu\) 不同。
