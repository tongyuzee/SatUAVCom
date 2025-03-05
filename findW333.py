import numpy as np
import matplotlib.pyplot as plt

class BeamformingOptimizer:
    def __init__(self, N, K, SNR_dB, sigma2=1, iter_max=100):
        """
        Initialize the BeamformingOptimizer with system parameters.

        Parameters:
        - N (int): Number of base station antennas
        - K (int): Number of users
        - SNR_dB (float): Signal-to-noise ratio in dB
        - sigma2 (float): Noise variance (default=1)
        - iter_max (int): Maximum number of iterations (default=100)
        """
        self.N = N
        self.K = K
        self.SNR_dB = SNR_dB
        self.sigma2 = sigma2
        self.P = 10 ** (SNR_dB / 10) * sigma2  # Transmit power constraint
        self.iter_max = iter_max
        self.h = self.generate_channel()  # Channel matrix
        self.w = self.initialize_beamformers()  # Beamforming vectors
        self.rate_history = []  # Store sum rate per iteration

    def generate_channel(self):
        """
        Generate the channel matrix h of size N x K with complex Gaussian entries.
        """
        return np.sqrt(1 / 2) * (np.random.randn(self.N, self.K) + 1j * np.random.randn(self.N, self.K))

    def initialize_beamformers(self):
        """
        Initialize beamforming vectors w with random complex entries, normalized to power P.
        """
        w_ini = np.random.randn(self.N, self.K) + 1j * np.random.randn(self.N, self.K)
        n = np.trace(w_ini @ w_ini.conj().T).real  # Total power of initial w
        return np.sqrt(self.P) * (w_ini / np.sqrt(n))

    def generate_U(self):
        """
        Generate receive filters U for each user.
        """
        sum1 = sum(np.abs(np.dot(self.h[:, i].conj().T, self.w[:, i]))**2 for i in range(self.K))
        U = np.array([np.dot(self.h[:, i].conj().T, self.w[:, i]) / (sum1 + self.sigma2) 
                      for i in range(self.K)], dtype=complex)
        return U

    def generate_W(self):
        """
        Generate weights W for each user.
        """
        sum1 = sum(np.abs(np.dot(self.h[:, i].conj().T, self.w[:, i]))**2 for i in range(self.K))
        W = np.array([1 / (1 - (np.abs(np.dot(self.h[:, i].conj().T, self.w[:, i]))**2) / (sum1 + self.sigma2)) 
                      for i in range(self.K)])
        return W

    def generate_V(self, U, W):
        """
        Generate optimized beamforming vectors V using bisection to enforce power constraint.

        Parameters:
        - U (ndarray): Receive filters
        - W (ndarray): Weights
        """
        sum2 = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.K):
            sum2 += (U[i] * W[i] * np.conj(U[i])) * np.outer(self.h[:, i], self.h[:, i].conj())

        mu_min = 0
        mu_max = 10
        iter = 0
        iter_max = 100

        while True:
            mu1 = (mu_min + mu_max) / 2
            V_opt = np.zeros((self.N, self.K), dtype=complex)
            Pt = 0
            for i in range(self.K):
                V_opt[:, i] = np.linalg.solve(sum2 + mu1 * np.eye(self.N), self.h[:, i]) * (U[i] * W[i])
                Pt += np.real(np.dot(V_opt[:, i].conj().T, V_opt[:, i]))
            
            if Pt > self.P:
                mu_min = mu1
            else:
                mu_max = mu1
            
            iter += 1
            if abs(mu_max - mu_min) < 1e-5 or iter > iter_max:
                break
        
        mu = mu1
        print(f'求解最优mu共迭代{iter}次, mu*={mu}, P={Pt}')
        
        V = np.zeros((self.N, self.K), dtype=complex)
        for i in range(self.K):
            V[:, i] = np.linalg.solve(sum2 + mu * np.eye(self.N), self.h[:, i]) * (U[i] * W[i])
        
        return V

    def cal_R(self):
        """
        Calculate the sum rate R.
        """
        sum1 = sum(np.abs(np.dot(self.h[:, i].conj().T, self.w[:, i]))**2 for i in range(self.K))
        R = 0
        for i in range(self.K):
            interfere = sum1 - np.abs(np.dot(self.h[:, i].conj().T, self.w[:, i]))**2
            INR = self.sigma2 + interfere
            signal = np.abs(np.dot(self.h[:, i].conj().T, self.w[:, i]))**2
            R += np.log2(1 + signal / INR)
        return np.abs(R)

    def optimize(self):
        """
        Run the iterative optimization to maximize the sum rate.
        
        Returns:
        - rate_history (list): Sum rate at each iteration
        """
        iter = 0
        while True:
            R_pre = self.cal_R()
            self.rate_history.append(R_pre)
            U = self.generate_U()
            W = self.generate_W()
            self.w = self.generate_V(U, W)
            R = self.cal_R()
            iter += 1
            if abs(R - R_pre) < 1e-5 or iter > self.iter_max:
                break
        self.rate_history.append(R)
        print(f'求解和速率共迭代{iter}次')
        return self.rate_history

# Example usage
if __name__ == "__main__":
    N = 4      # Number of antennas
    K = 10     # Number of users
    SNR_dB = 20  # SNR in dB
    optimizer = BeamformingOptimizer(N, K, SNR_dB)
    rate_history = optimizer.optimize()

    # Plot the results
    plt.plot(rate_history, 'r-o')
    plt.ylabel('和速率 (Sum Rate)')
    plt.xlabel('迭代次数 (Iteration)')
    plt.show()