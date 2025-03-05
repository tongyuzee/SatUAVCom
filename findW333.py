import numpy as np
import matplotlib.pyplot as plt

# Parameter Settings
N = 4  # Number of base station transmit antennas
K = 10  # Number of users
SNR_dB = 20  # Signal-to-noise ratio in dB
sigma2 = 1  # Noise power
P = 10**(SNR_dB / 10) * sigma2  # Transmit power constraint
iter = 1  # Iteration counter
iter_max = 100  # Maximum number of iterations

# Channel Matrix Generation
# Generate K channel vectors h, each an N-dimensional complex vector
h = np.sqrt(0.5) * (np.random.randn(N, K) + 1j * np.random.randn(N, K))

# Initialize Precoding Vectors
w_ini1 = np.random.randn(N, K) + 1j * np.random.randn(N, K)
n = np.trace(w_ini1 @ w_ini1.conj().T)  # Trace of w_ini1 * w_ini1^H
w_ini = np.sqrt(P) * (w_ini1 / np.sqrt(n))  # Normalize to satisfy power constraint
w = w_ini.copy()  # Current precoding matrix

# Function Definitions

def cal_R(h, w, sigma2, K):
    """Calculate the sum rate."""
    sum1 = 0
    for i in range(K):
        sum1 += np.vdot(h[:, i], w[:, i]) * np.vdot(w[:, i], h[:, i])
    interfere = np.zeros(K)
    R = 0
    for i in range(K):
        interfere[i] = sum1 - np.vdot(h[:, i], w[:, i]) * np.vdot(w[:, i], h[:, i])
        INR = sigma2 + interfere[i]  # Interference plus noise
        sinal = np.vdot(h[:, i], w[:, i]) * np.vdot(w[:, i], h[:, i])  # Desired signal
        R += np.log2(1 + (sinal / INR))  # Rate for user i
    return np.abs(R)

def generate_U(h, w, sigma2, K):
    """Generate receive filters U."""
    U = np.zeros(K, dtype=complex)
    sum1 = 0
    for i in range(K):
        sum1 += np.vdot(h[:, i], w[:, i]) * np.vdot(w[:, i], h[:, i])
    for i in range(K):
        U[i] = np.vdot(h[:, i], w[:, i]) / (sum1 + sigma2)
    return U

def generate_W(h, w, sigma2, K):
    """Generate weights W."""
    W = np.zeros(K)
    sum1 = 0
    for i in range(K):
        sum1 += np.vdot(h[:, i], w[:, i]) * np.vdot(w[:, i], h[:, i])
    for i in range(K):
        temp = np.vdot(w[:, i], h[:, i]) * (1 / (sum1 + sigma2)) * np.vdot(h[:, i], w[:, i])
        W[i] = 1 / (1 - temp)
    return W

def generate_V(h, U, W, N, K, P):
    """Generate precoding vectors V."""
    sum2 = np.zeros((N, N), dtype=complex)
    for i in range(K):
        sum2 += np.outer(h[:, i], h[:, i].conj()) * U[i] * W[i] * np.conj(U[i])
    
    # Binary search for optimal mu
    mu_max = 10
    mu_min = 0
    iter = 0
    iter_max = 100
    
    while True:
        mu1 = (mu_min + mu_max) / 2
        Pt = 0
        V_opt = np.zeros((N, K), dtype=complex)
        for i in range(K):
            A = sum2 + mu1 * np.eye(N)
            b = h[:, i] * U[i] * W[i]
            V_opt[:, i] = np.linalg.solve(A, b)
            Pt += np.real(np.trace(np.outer(V_opt[:, i], V_opt[:, i].conj())))
        
        if Pt > P:
            mu_min = mu1
        else:
            mu_max = mu1
        
        iter += 1
        
        if abs(mu_max - mu_min) < 1e-5 or iter > iter_max:
            break
    
    mu = mu1
    print(f'求解最优mu共迭代{iter}次,mu*={mu},P={Pt}')
    
    V = np.zeros((N, K), dtype=complex)
    for i in range(K):
        A = sum2 + mu * np.eye(N)
        b = h[:, i] * U[i] * W[i]
        V[:, i] = np.linalg.solve(A, b)
    
    return V

# Iterative Algorithm
rate = []
while True:
    R_pre = cal_R(h, w, sigma2, K)  # Previous sum rate
    rate.append(R_pre)
    U = generate_U(h, w, sigma2, K)  # Update receive filters
    W_sup = generate_W(h, w, sigma2, K)  # Update weights
    w = generate_V(h, U, W_sup, N, K, P)  # Update precoding vectors
    R = cal_R(h, w, sigma2, K)  # Current sum rate
    iter += 1
    if abs(R - R_pre) < 1e-5 or iter > iter_max:  # Convergence check
        break

rate.append(R)  # Append final rate
print(f'求解和速率共迭代{iter}次')

# Plot Results
plt.plot(rate, 'r-o')
plt.ylabel('和速率')
plt.xlabel('迭代次数')
plt.show()