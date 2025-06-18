import scipy
import numpy as np
scipy.pi  = np.pi
scipy.sin = np.sin
scipy.cos = np.cos
from scipy.optimize import minimize
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pymcmcstat.MCMC import MCMC

# Load observed data 
csv_path = "./Cholera-SIBR-Model/data/real_data.csv"
data     = pd.read_csv(csv_path)
t_data   = np.arange(len(data))                  
I_obs    = data["Total Cases (Weekly)"].values

# reshape into column vectors for pymcmcstat
t_col = t_data.reshape(-1, 1)
I_col = I_obs .reshape(-1, 1)

# Fixed (non‐fitted) parameters
fixed_params = {
    "phi0":  0.015,
    "tau":   0.5,
    "k1":    1e6,
    "gamma": 0.2,
    "k":     1e5,
    "xi":    10,
    "beta":  1/30,
    "alpha": 43.5,
}

# SIBR ODE system
def sibr_model(y, t, sigma_e, sigma_h, omega, delta, phi1, a):
    S, I, B, R = y
    phi0 = fixed_params["phi0"]
    Phi  = phi0 + (phi1 - phi0) * (a / (I + a))
    inf_force = sigma_e * B/(fixed_params["k1"] + B) + sigma_h * I

    dS = fixed_params["pi"] + fixed_params["tau"]*R \
         - (1 - omega)*inf_force*S \
         - fixed_params["alpha"]*S
    dI = (1 - omega)*inf_force*S \
         - Phi*I \
         - fixed_params["alpha"]*I
    dB = fixed_params["gamma"]*B*(1 - B/fixed_params["k"]) \
         + fixed_params["xi"]*I \
         - (fixed_params["beta"] + delta)*B
    dR = Phi*I \
         - (fixed_params["tau"] + fixed_params["alpha"])*R

    return [dS, dI, dB, dR]

# Initial‐condition guess
y0 = [197112, 5646, 21259, 118]   # [S(0), I(0), B(0), R(0)]


def ssq_theta(theta):
    # ensure pi is set for the MLE step, too
    S0, I0, B0, R0 = y0
    fixed_params["pi"] = fixed_params["alpha"] * (S0 + I0 + R0)

    sol = odeint(sibr_model, y0, t_data, args=tuple(theta))
    return np.mean((sol[:,1] - I_obs)**2)

res_theta = minimize(ssq_theta,
                     x0=[0.058,1.4e-5,0.86,5.0,0.9,14.0],
                     bounds=[(0.055,0.082),
                             (1.08e-5,1.7e-5),
                             (0.83,0.89),
                             (3.2,5.5),
                             (0.078,0.098),
                             (10,20)],
                     method='L-BFGS-B')
theta_mle = res_theta.x

# Sum‐of‐squares objective for pymcmcstat
def ssq_fun(theta, data, custom=None):
    σe, σh, ω, δ, ϕ1, a = theta
    S0, B0, R0 = y0[0], y0[2], y0[3]
    fixed_params["pi"] = fixed_params["alpha"] * (S0 + B0 + R0)

    # simulate
    sol = odeint(sibr_model, y0, t_data,
                 args=(σe, σh, ω, δ, ϕ1, a))
    I_pred = sol[:,1]

    obs = data.ydata[0].flatten()
    resid = obs - I_pred
    return np.sum(resid**2)

# Build and run the DRAM sampler 
mcmc = MCMC()

# supply data
mcmc.data.add_data_set(x=t_col, y=I_col)

mcmc.model_settings.define_model_settings(sos_function=ssq_fun)

mcmc.simulation_options.define_simulation_options(
    nsimu=1000,
    method='dram'
)

mcmc.parameters.add_model_parameter(name="sigma_e", theta0=theta_mle[0],  minimum=0.055,    maximum=0.082)
mcmc.parameters.add_model_parameter(name="sigma_h", theta0=theta_mle[1], minimum=1.08e-5,    maximum=1.7e-5)
mcmc.parameters.add_model_parameter(name="omega",   theta0=theta_mle[2],  minimum=0.83,    maximum=0.89)
mcmc.parameters.add_model_parameter(name="delta",   theta0=theta_mle[3],   minimum=3.2,    maximum=5.5)
mcmc.parameters.add_model_parameter(name="phi1",    theta0=theta_mle[4],   minimum=0.078,    maximum=0.098)
mcmc.parameters.add_model_parameter(name="a",       theta0=theta_mle[5],  minimum=10,    maximum=20)

mcmc.run_simulation()

chain = mcmc.simulation_results.results['chain'] 

param_means  = np.mean(chain, axis=0)
param_ci_low, param_ci_high = np.percentile(chain, [2.5, 97.5], axis=0)
table3 = pd.DataFrame({
    'Parameter': ['σₑ','σₕ','ω','δ','ϕ₁','a'],
    'Mean':      param_means,
    '2.5 % CI':  param_ci_low,
    '97.5 % CI': param_ci_high
})
print("\n 95 % CIs")
print(table3.to_string(index=False))

idx    = np.random.choice(chain.shape[0], 300, replace=False)
I_sims = np.vstack([
    odeint(sibr_model, y0, t_data, args=tuple(chain[i]))[:,1]
    for i in idx
])
lower  = np.percentile(I_sims,  2.5, axis=0)
upper  = np.percentile(I_sims, 97.5, axis=0)
median = np.percentile(I_sims, 50, axis=0)

plt.figure(figsize=(8,4))
plt.fill_between(t_data, lower, upper, alpha=0.3, label='95 % CI')
plt.plot(t_data, median, 'b-', label='Median fit')
plt.plot(t_data, I_obs,   'ro', label='Observed')
plt.xlabel("Time (weeks)")
plt.ylabel("Weekly cases")
plt.legend()
plt.title("95 % CI")
plt.tight_layout()
plt.show()

param_names = ['σₑ','σₕ','ω','δ','ϕ₁','a']
n_iter = chain.shape[0]
for i, pname in enumerate(param_names):
    plt.figure()
    plt.scatter(np.arange(n_iter), chain[:, i], s=10)
    plt.xlabel("Length of Chain")
    plt.ylabel(pname)
    plt.tight_layout()
    plt.show()

