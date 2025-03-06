
# WMMSE Algorithm Derivation

下面给出 WMMSE 算法从速率最大化问题到加权均方误差问题的详细推导过程。为了便于说明，我们以单载波、多用户干扰信道为例（注意这里的推导思路同样适用于 RIS 辅助的 LEO–UAV 系统，只是在信道模型上会更复杂）。

---

## 1. 原始速率最大化问题

对于第 \( u \) 个用户，其接收信号模型可以写为
\[
y_u = \underbrace{\boldsymbol{h}_u^T\boldsymbol{w}_u s_u}_{\text{目标信号}} + \underbrace{\sum_{v\neq u}\boldsymbol{h}_u^T\boldsymbol{w}_v s_v}_{\text{干扰}} + n_u,
\]
其中  

- \(\boldsymbol{w}_u\) 为用户 \( u \) 的波束成形向量，  
- \(\boldsymbol{h}_u\) 为用户 \( u \) 的有效信道，  
- \(s_u\) 为发送信号，  
- \(n_u\) 为噪声。  

用户 \( u \) 的信噪比（SINR）为
\[
\gamma_u = \frac{|\boldsymbol{h}_u^T\boldsymbol{w}_u|^2}{\sum_{v\neq u} |\boldsymbol{h}_u^T\boldsymbol{w}_v|^2 + \sigma^2}.
\]
因此，该用户的速率为
\[
R_u = \log_2 (1+\gamma_u).
\]

全系统的速率最大化问题写为
\[
\max_{\{\boldsymbol{w}_u\}} \sum_{u=1}^{U} \log_2 (1+\gamma_u) \quad \text{s.t.} \quad \sum_{u=1}^{U}\|\boldsymbol{w}_u\|^2 \le P_{\max}.
\]
由于 \(\gamma_u\) 是 \(\boldsymbol{w}_u\) 的非凸函数，上述问题是非凸优化问题。

---

## 2. 引入均衡器与均方误差

引入每个用户的线性接收均衡器 \(g_u\)，对接收到的信号进行处理，估计出的信号为
\[
\hat{s}_u = g_u y_u.
\]
定义第 \( u \) 个用户的均方误差（MSE）为
\[
e_u = \mathbb{E}\left\{|s_u - g_u y_u|^2\right\}.
\]

将 \(y_u\) 的表达式代入并展开，得到
\[
e_u = \underbrace{\left|1 - g_u\boldsymbol{h}_u^T\boldsymbol{w}_u\right|^2}_{\text{误差项1}} + \underbrace{\sum_{v\neq u}|g_u\boldsymbol{h}_u^T\boldsymbol{w}_v|^2}_{\text{干扰项}} + |g_u|^2\sigma^2.
\]

对于固定的波束成形向量 \(\{\boldsymbol{w}_u\}\) 与噪声水平，针对每个 \(u\) 的 \(g_u\) 可求解使 \(e_u\) 最小的最优均衡器，这个最优均衡器为
\[
g_u^\star = \frac{\boldsymbol{h}_u^T\boldsymbol{w}_u}{\sum_{v=1}^{U} |\boldsymbol{h}_u^T\boldsymbol{w}_v|^2 + \sigma^2}.
\]

将 \(g_u^\star\) 代入 \(e_u\) 后，经过一些推导可以证明（这是信息论中的经典结果）：
\[
e_u^\star = \frac{1}{1+\gamma_u}.
\]
这就建立了最小均方误差与 SINR 之间的关系，是 WMMSE 方法的核心所在。

---

## 3. 构造加权均方误差目标函数

为了将速率最大化问题转化为等价的 MSE 优化问题，引入权重 \(\lambda_u > 0\) 并构造下列目标函数：
\[
J = \sum_{u=1}^{U} \left(\lambda_u e_u - \log \lambda_u\right).
\]

在给定 \(\{\boldsymbol{w}_u\}\) 和 \(g_u\) 的条件下，对于每个用户 \(u\)，固定其他变量，优化 \(\lambda_u\) 可得最优解为
\[
\lambda_u^\star = \frac{1}{e_u^\star}.
\]

将最优 \(\lambda_u^\star\) 代入目标函数，得到
\[
J^\star = \sum_{u=1}^{U}\left(\frac{e_u^\star}{e_u^\star} - \log\frac{1}{e_u^\star}\right) = \sum_{u=1}^{U} \left(1 + \log e_u^\star\right).
\]
又因为 \(e_u^\star = \frac{1}{1+\gamma_u}\)，所以
\[
1+\log e_u^\star = 1 - \log(1+\gamma_u).
\]
因此
\[
J^\star = U - \sum_{u=1}^{U} \log(1+\gamma_u).
\]

最小化 \(J\) 与最大化 \(\sum_{u}\log(1+\gamma_u)\) 等价（常数 \(U\) 不影响最优解），从而原问题转化为
\[
\min_{\{\boldsymbol{w}_u, g_u, \lambda_u\}} \sum_{u=1}^{U} \left(\lambda_u e_u - \log \lambda_u\right) \quad \text{s.t.} \quad \sum_{u=1}^{U}\|\boldsymbol{w}_u\|^2 \le P_{\max}.
\]

---

## 4. 交替优化求解步骤

由于上述目标函数在 \(\boldsymbol{w}_u\)、\(g_u\) 以及 \(\lambda_u\) 上均非凸，但对每个变量单独固定其他变量时问题是凸的，因此可采用交替优化（Block Coordinate Descent）方法进行求解。具体步骤如下：

1. **初始化**：随机初始化所有波束成形向量 \(\{\boldsymbol{w}_u\}\)。

2. **更新均衡器 \(g_u\)**  
   对于每个 \(u\)，在固定 \(\{\boldsymbol{w}_u\}\) 的情况下，最优均衡器为
   \[
   g_u^\star = \frac{\boldsymbol{h}_u^T\boldsymbol{w}_u}{\sum_{v=1}^{U} |\boldsymbol{h}_u^T\boldsymbol{w}_v|^2 + \sigma^2}.
   \]

3. **更新权重 \(\lambda_u\)**  
   利用最优均方误差 \(e_u^\star\)（通过 \(g_u^\star\) 计算得到），更新权重为
   \[
   \lambda_u^\star = \frac{1}{e_u^\star}.
   \]

4. **更新波束成形向量 \(\boldsymbol{w}_u\)**  
   在固定 \(g_u\) 和 \(\lambda_u\) 后，对每个 \(u\) 求解下列凸优化问题：
   \[
   \min_{\{\boldsymbol{w}_u\}} \sum_{u=1}^{U} \lambda_u e_u \quad \text{s.t.} \quad \sum_{u=1}^{U}\|\boldsymbol{w}_u\|^2 \le P_{\max}.
   \]
   这一步通常需要借助拉格朗日乘子法或二次规划求解。

5. **迭代**：重复步骤 2–4，直到目标函数收敛或达到预设迭代次数。

---

## 5. 总结

- **关键关系**：  
  通过引入均衡器 \(g_u\) 得到 MSE \(e_u\)，并证明最优 MSE 与 SINR 的关系为  
  \[
  e_u^\star = \frac{1}{1+\gamma_u}.
  \]
- **目标函数构造**：  
  构造加权 MSE 目标函数  
  \[
  J = \sum_{u=1}^{U} (\lambda_u e_u - \log \lambda_u)
  \]
  并证明最小化 \(J\) 与最大化 \(\sum_{u}\log(1+\gamma_u)\) 等价。
- **交替优化**：  
  分别对 \(g_u\)、\(\lambda_u\) 和 \(\boldsymbol{w}_u\) 进行交替更新，直至收敛。

这一推导过程为 WMMSE 算法提供了理论基础，其核心在于将原来的速率最大化问题转换为一个关于均方误差的优化问题，再通过引入权重使得两者目标一致，进而利用交替优化方法求得近似全局最优解。

以上即为 WMMSE 算法的详细推导步骤。

---

在通信领域中，WMMSE（Weighted Minimum Mean Squared Error， 加权最小均方误差）算法是一种常用的优化方法，特别是在多用户MIMO（Multiple-Input Multiple-Output）系统中的信号检测和波束成形设计中。你的公式看起来像是描述了一个多用户系统中第 \( k \) 个用户的接收信号模型：

\[ y_k = H_k w_k s_k + \sum_{k' \neq k} H_k w_{k'} s_{k'} + n \]

其中：

- \( y_k \) 是第 \( k \) 个用户的接收信号；
- \( H_k \) 是第 \( k \) 个用户的信道矩阵；
- \( w_k \) 是第 \( k \) 个用户的预编码向量（或波束成形向量）；
- \( s_k \) 是第 \( k \) 个用户的发送符号；
- \( \sum_{k' \neq k} H_k w_{k'} s_{k'} \) 表示其他用户 \( k' \neq k \) 对第 \( k \) 个用户的干扰；
- \( n \) 是加性高斯白噪声。

在这种模型下，WMMSE的目标是通过优化预编码向量 \( w_k \) 和接收滤波器（通常假设已知或可设计），来最小化加权均方误差，同时考虑多用户干扰。我们将逐步推导WMMSE算法的优化过程。

---

### 1. 系统模型和目标

假设第 \( k \) 个用户使用一个接收滤波器 \( u_k \) 来估计发送符号 \( s_k \)，其估计值为：
\[ \hat{s}_k = u_k^H y_k \]
其中 \( u_k^H \) 是 \( u_k \) 的共轭转置（Hermitian transpose）。

将信号模型代入：
\[ \hat{s}_k = u_k^H H_k w_k s_k + u_k^H \sum_{k' \neq k} H_k w_{k'} s_{k'} + u_k^H n \]

均方误差（MSE）定义为：
\[ \epsilon_k = E[|\hat{s}_k - s_k|^2] \]

假设：

- \( E[s_k s_k^H] = 1 \)（发送符号功率归一化）；
- \( E[s_k s_{k'}^H] = 0 \)（不同用户符号不相关，\( k \neq k' \)）；
- \( n \sim \mathcal{CN}(0, \sigma^2 I) \)（噪声为零均值高斯噪声，方差为 \( \sigma^2 \)）。

展开MSE：
\[ \epsilon_k = E[|u_k^H y_k - s_k|^2] \]
\[ = E[|u_k^H H_k w_k s_k + u_k^H \sum_{k' \neq k} H_k w_{k'} s_{k'} + u_k^H n - s_k|^2] \]

计算期望：
\[ \epsilon_k = |u_k^H H_k w_k - 1|^2 + \sum_{k' \neq k} |u_k^H H_k w_{k'}|^2 + \sigma^2 \|u_k\|^2 \]

- 第一项 \( |u_k^H H_k w_k - 1|^2 \) 是信号失真；
- 第二项 \( \sum_{k' \neq k} |u_k^H H_k w_{k'}|^2 \) 是多用户干扰；
- 第三项 \( \sigma^2 \|u_k\|^2 \) 是噪声功率。

WMMSE的目标是最小化加权MSE的总和：
\[ \min_{u_k, w_k} \sum_{k} \alpha_k \epsilon_k \]
其中 \( \alpha_k \) 是第 \( k \) 个用户的权重（通常与优先级或QoS相关）。

---

### 2. WMMSE优化步骤

WMMSE算法通过交替优化 \( u_k \)（接收滤波器）、\( w_k \)（预编码向量）和权重（通常引入拉格朗日乘子或等效变换）来解决问题。由于这是一个非凸问题，我们将其分解为多个凸子问题。

#### (1) 优化接收滤波器 \( u_k \)

固定 \( w_k \)（所有用户的预编码向量），对 \( u_k \) 求偏导数，最小化 \( \epsilon_k \)。

MSE可写为：
\[ \epsilon_k = (u_k^H H_k w_k - 1)(u_k^H H_k w_k - 1)^H + \sum_{k' \neq k} u_k^H H_k w_{k'} w_{k'}^H H_k^H u_k + \sigma^2 u_k^H u_k \]

定义协方差矩阵：
\[ R_k = H_k \left( \sum_{k'} w_{k'} w_{k'}^H \right) H_k^H + \sigma^2 I \]
其中 \( \sum_{k'} w_{k'} w_{k'}^H \) 是所有用户的预编码协方差。

那么：
\[ \epsilon_k = u_k^H R_k u_k - u_k^H H_k w_k - w_k^H H_k^H u_k + 1 \]

对 \( u_k \) 求导并令其为零：
\[ \frac{\partial \epsilon_k}{\partial u_k^*} = R_k u_k - H_k w_k = 0 \]
\[ u_k = R_k^{-1} H_k w_k \]

这是最优的MMSE接收滤波器。

#### (2) 优化预编码向量 \( w_k \)

固定 \( u_k \)，最小化 \( \sum_k \alpha_k \epsilon_k \)，通常需要引入功率约束：
\[ \sum_k \|w_k\|^2 \leq P_{\text{total}} \]

这是一个带约束的优化问题，使用拉格朗日乘子法。拉格朗日函数为：
\[ L = \sum_k \alpha_k \epsilon_k + \mu \left( \sum_k \|w_k\|^2 - P_{\text{total}} \right) \]

将 \( u_k = R_k^{-1} H_k w_k \) 代入 \( \epsilon_k \)，MSE简化为：
\[ \epsilon_k = 1 - w_k^H H_k^H R_k^{-1} H_k w_k \]

目标函数变为：
\[ \sum_k \alpha_k (1 - w_k^H H_k^H R_k^{-1} H_k w_k) + \mu \left( \sum_k \|w_k\|^2 - P_{\text{total}} \right) \]

对 \( w_k \) 求导：
\[ \frac{\partial L}{\partial w_k^*} = -\alpha_k H_k^H R_k^{-1} H_k w_k + \mu w_k = 0 \]
\[ w_k = \frac{\alpha_k}{\mu} H_k^H R_k^{-1} H_k w_k \]

这需要迭代求解，因为 \( R_k \) 依赖于所有 \( w_k \)。

#### (3) 更新权重 \( \alpha_k \)

WMMSE算法的一个关键特性是将MSE与信干噪比（SINR）联系起来。SINR定义为：
\[ \text{SINR}_k = \frac{|u_k^H H_k w_k|^2}{\sum_{k' \neq k} |u_k^H H_k w_{k'}|^2 + \sigma^2 \|u_k\|^2} \]

最优时，\( \alpha_k = \frac{1}{\epsilon_k} \)（通过KKT条件导出），这将MSE最小化问题转化为等效的SINR最大化问题。

---

### 3. 算法流程

1. **初始化**：随机选择初始 \( w_k \)，满足功率约束。
2. **更新 \( u_k \)**：计算 \( R_k \)，然后 \( u_k = R_k^{-1} H_k w_k \)。
3. **更新 \( \alpha_k \)**：计算 \( \epsilon_k \)，设置 \( \alpha_k = \frac{1}{\epsilon_k} \)。
4. **更新 \( w_k \)**：解带功率约束的优化问题，更新预编码向量。
5. **迭代**：重复步骤2-4直到收敛。

---

### 4. 总结

WMMSE通过交替优化 \( u_k \)、\( w_k \) 和 \( \alpha_k \)，将复杂的多用户干扰管理问题转化为可解的子问题，最终收敛到局部最优解。这个推导基于你的信号模型 \( y_k = H_k w_k s_k + \sum_{k' \neq k} H_k w_{k'} s_{k'} + n \)，适用于多用户MIMO系统的波束成形设计。如果需要更具体的数值示例或代码实现，请告诉我！
