import numpy as np


###############
# Unified SAM #
###############
# Some code taken from Oikonomou et al, 2025

def Unified_SAM(loss, trials, record_f, x0, gamma, rho, lambd, bs=1):
    # Unified SAM
    # e^t = x^t + \rho_t * (1-\lambda_t + \lambda_t/|\nabla f_{i_t}(x^t)|) * \nabla f_{i_t}(x^t)
    # x^{t+1} = x^t - \gamma_t * \nabla f_{i_t}(e^t)

    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    T = record_f[-1]

    for trial in range(trials):
        x = [x0 for i in range(T+1)]
        e = [x0 for i in range(T)]
        counter = 1

        for t in range(T):
            i_t = np.random.choice(a=range(loss.n), size=bs)
            g_x = loss.grad(x[t], i_t)
            
            e[t] = x[t] + rho * (1 - lambd + lambd/np.linalg.norm(g_x)) * g_x
            g_e = loss.grad(e[t], i_t)

            x[t+1] = x[t] - gamma * g_e

            if t+1 in record_f:
                f[counter, trial] = loss.func(x[t+1], full_batch)
                counter += 1

    name = r'Unified SAM, $\lambda='+str(lambd)+r'$, $\rho='+'{:.3f}'.format(rho)+r'$, $\gamma='+'{:.3f}'.format(gamma)+r'$'
    return name, f

def Unified_SAM_SPS(loss, trials, record_f, x0, rho, lambd, f_star, bs=1):
    # Unified SAM SPS
    # e^t = x^t + \rho_t * (1-\lambda_t + \lambda_t/|\nabla f_{i_t}(x^t)|) * \nabla f_{i_t}(x^t)
    # \gamma_t = \frac{f_{i_t}(e^t)-f_{i_t}^*-\langle\nabla f_{i_t}(e^t), e^t-x^t\rangle}{\|\nabla f_{i_t}(e^t)\|^2}
    # x^{t+1} = x^t - \gamma_t * \nabla f_{i_t}(e^t)

    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    T = record_f[-1]

    for trial in range(trials):
        x = [x0 for i in range(T+1)]
        e = [x0 for i in range(T)]
        gamma = [x0 for i in range(T)]
        counter = 1

        for t in range(T):
            i_t = np.random.choice(a=range(loss.n), size=bs)
            g_x = loss.grad(x[t], i_t)

            e[t] = x[t] + rho * (1 - lambd + lambd/np.linalg.norm(g_x)) * g_x
            g_e = loss.grad(e[t], i_t)
            
            gamma[t] = (loss.func(e[t], i_t)-f_star-np.inner(g_e, e[t]-x[t]))/np.linalg.norm(g_e)**2
            x[t+1] = x[t] - gamma[t] * g_e

            if t+1 in record_f:
                f[counter, trial] = loss.func(x[t+1], full_batch)
                counter += 1

    name = r'Unified SAM SPS, $\lambda='+str(lambd)+r'$, $\rho='+'{:.3f}'.format(rho)+r'$'
    return name, f

def Unified_SAM_SPS_max(loss, trials, record_f, x0, rho, lambd, f_star, gamma_b, bs=1):
    # Unified SAM SPS
    # e^t = x^t + \rho_t * (1-\lambda_t + \lambda_t/|\nabla f_{i_t}(x^t)|) * \nabla f_{i_t}(x^t)
    # \gamma_t = \frac{f_{i_t}(e^t)-f_{i_t}^*-\langle\nabla f_{i_t}(e^t), e^t-x^t\rangle}{\|\nabla f_{i_t}(e^t)\|^2}
    # x^{t+1} = x^t - \gamma_t * \nabla f_{i_t}(e^t)

    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    T = record_f[-1]

    for trial in range(trials):
        x = [x0 for i in range(T+1)]
        e = [x0 for i in range(T)]
        gamma = [x0 for i in range(T)]
        counter = 1

        for t in range(T):
            i_t = np.random.choice(a=range(loss.n), size=bs)
            g_x = loss.grad(x[t], i_t)

            e[t] = x[t] + rho * (1 - lambd + lambd/np.linalg.norm(g_x)) * g_x
            g_e = loss.grad(e[t], i_t)
            
            gamma[t] = min((loss.func(e[t], i_t)-f_star-np.inner(g_e, e[t]-x[t]))/np.linalg.norm(g_e)**2,gamma_b)
            x[t+1] = x[t] - gamma[t] * g_e

            if t+1 in record_f:
                f[counter, trial] = loss.func(x[t+1], full_batch)
                counter += 1

    name = r'Unified SAM SPS, $\lambda='+str(lambd)+r'$, $\rho='+'{:.3f}'.format(rho)+r'$'
    return name, f

def USAM_andr(loss, trials, record_f, x0, bs=1):
    # USAM from Andriushchenko et al
    # e^t = x^t + \rho_t * (1-\lambda_t + \lambda_t/|\nabla f_{i_t}(x^t)|) * \nabla f_{i_t}(x^t)
    # x^{t+1} = x^t - \gamma_t * \nabla f_{i_t}(e^t)
    # \gamma_t = \min\{(8t+4)/(3*mu*(t+1)^2),1/(2*L_max)\}
    # \rho_t = \sqrt{\gamma_t/L_max}
    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    T = record_f[-1]

    for trial in range(trials):
        x = [x0 for i in range(T+1)]
        e = [x0 for i in range(T)]
        counter = 1

        for t in range(T):
            i_t = np.random.choice(a=range(loss.n), size=bs)
            g_x = loss.grad(x[t], i_t)
            
            L_max = loss.L_max
            gamma_t = min((8*t+4)/(3*loss.mu*(t+1)**2),1/(2*L_max))
            rho_t = (gamma_t/L_max)**0.5
            # gamma_t = (8*t+4)/(3*loss.mu*(t+1)**2)
            # rho_t = (gamma_t/L_max)**0.5
            e[t] = x[t] + rho_t * g_x
            g_e = loss.grad(e[t], i_t)

            x[t+1] = x[t] - gamma_t * g_e

            if t+1 in record_f:
                f[counter, trial] = loss.func(x[t+1], full_batch)
                counter += 1

    name = r'USAM Andriushchenko Stochastic'
    return name, f
