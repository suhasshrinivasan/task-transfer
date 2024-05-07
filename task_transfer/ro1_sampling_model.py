import torch
import torch.distributions as dist


class R01SamplingModel:
    """
    A class for the R01 orientation discrimination task based sampling generative model,
    designed to estimate the posterior distribution of C, g, and x given I and T.

    Attributes:
        n_g (int): Number of g variables.
        n_x (int): Number of x variables.
        sigma_C_T (float): Standard deviation parameter for C and T relation.
        sigma_x (float): Standard deviation for x sampling.
        A (torch.Tensor): Matrix linking x to I.
        sigma_I (float): Standard deviation of the observational noise in I.
        C (torch.Tensor): Current sample of C.
        g (torch.Tensor): Current samples of g variables.
        x (torch.Tensor): Current samples of x variables.
    """

    def __init__(self, n_g, n_x, sigma_C_T, sigma_x, A, sigma_I):
        """
        Initializes the GibbsSampler with specified parameters and initial random samples for C, g, and x.
        """
        self.n_g = n_g
        self.n_x = n_x
        self.sigma_C_T = sigma_C_T
        self.sigma_x = sigma_x
        self.A = torch.tensor(A, dtype=torch.float32)
        self.sigma_I = sigma_I

        # Initial random samples
        self.C = torch.bernoulli(torch.tensor([0.5]))
        self.g = torch.bernoulli(torch.tensor([0.5] * n_g))
        self.x = torch.bernoulli(torch.tensor([0.5] * n_x))

    def sample_C(self, T):
        """Samples a new C based on current values of g, x, and given T."""
        log_probs = torch.zeros(2)
        for C in range(2):
            theta_C_T = torch.full((self.n_g,), C * T, dtype=torch.float32)
            probs_g = torch.exp(-0.5 / self.sigma_C_T**2 * (self.g - theta_C_T).pow(2))
            log_probs[C] = dist.Bernoulli(probs_g).log_prob(self.g).sum()
        self.C = dist.Categorical(logits=log_probs).sample()

    def sample_g(self, T):
        """Samples new g values based on current values of C, x, and given T."""
        theta_C_T = self.C.float() * T
        for i in range(self.n_g):
            log_probs = torch.zeros(2)
            for gi in range(2):
                g_temp = self.g.clone()
                g_temp[i] = gi
                probs_g = torch.exp(
                    -0.5 / self.sigma_C_T**2 * (g_temp - theta_C_T) ** 2
                )
                log_prob_g = dist.Bernoulli(probs_g).log_prob(g_temp).sum()
                probs_x = torch.exp(
                    -0.5 / self.sigma_x**2 * (self.x - g_temp.float()) ** 2
                )
                log_prob_x = dist.Bernoulli(probs_x).log_prob(self.x).sum()
                log_probs[gi] = log_prob_g + log_prob_x
            self.g[i] = dist.Categorical(logits=log_probs).sample()

    def sample_x(self):
        """Samples new x values based on current values of g and observational model I."""
        for i in range(self.n_x):
            log_probs = torch.zeros(2)
            for xi in range(2):
                x_temp = self.x.clone()
                x_temp[i] = xi
                probs_x = torch.exp(
                    -0.5 / self.sigma_x**2 * (x_temp - self.g.float()).pow(2)
                )
                log_prob_x = dist.Bernoulli(probs_x).log_prob(x_temp).sum()
                log_prob_I = (
                    dist.Normal(self.A.matmul(x_temp.float()), self.sigma_I)
                    .log_prob(self.I)
                    .sum()
                )
                log_probs[xi] = log_prob_x + log_prob_I
            self.x[i] = dist.Categorical(logits=log_probs).sample()

    def run_sampling(self, T, I, iterations=1000):
        """
        Executes the Gibbs sampling algorithm over a specified number of iterations.

        Args:
            T (torch.Tensor): The temperature parameter influencing the distribution of C.
            I (torch.Tensor): Observational data influencing the distribution of x.
            iterations (int): Number of iterations for the Gibbs sampling.
        """
        self.I = torch.tensor(I, dtype=torch.float32)
        for _ in range(iterations):
            self.sample_C(T)
            self.sample_g(T)
            self.sample_x()


# # Example usage:
# n_g = 10
# n_x = 20
# sigma_C_T = 1.0
# sigma_x = 1.0
# A = np.random.normal(size=(100, n_x))
# sigma_I = 1.0
# I = np.random.normal(size=100)
# T = 0.5

# sampler = GibbsSampler(n_g, n_x, sigma_C_T, sigma_x, A, sigma_I)
# sampler.run_sampling(torch.tensor([T]), torch.tensor(I))
