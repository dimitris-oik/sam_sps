import numpy as np



##########
# AdaSAM #
##########
def AdaSAM(loss, trials, record_f, x0, gamma, rho, beta1, beta2, eps=1e-8, bs=1):
    # AdaSAM: SAM with adaptive learning rate and momentum acceleration
    # s_t   = \nabla f_B(x_t)
    # \delta_t = \rho * s_t / ||s_t||
    # g_t   = \nabla f_B(x_t + \delta_t)
    # m_t   = \beta_1 * m_{t-1} + (1-\beta_1) * g_t
    # v_t   = \beta_2 * v_{t-1} + (1-\beta_2) * g_t \odot g_t
    # \hat{v}_t = max(\hat{v}_{t-1}, v_t)
    # \eta_t = 1 / sqrt(\hat{v}_t)
    # x_{t+1} = x_t - \gamma * m_t \odot \eta_t

    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    T = record_f[-1]

    for trial in range(trials):
        x = [x0 for i in range(T+1)]
        m = np.zeros_like(x0)
        v = np.full_like(x0, eps**2, dtype=float)
        v_hat = np.full_like(x0, eps**2, dtype=float)
        counter = 1

        for t in range(T):
            i_t = np.random.choice(a=range(loss.n), size=bs)
            s_t = loss.grad(x[t], i_t)

            delta_t = rho * s_t / np.linalg.norm(s_t)
            g_t = loss.grad(x[t] + delta_t, i_t)

            m = beta1 * m + (1 - beta1) * g_t
            v = beta2 * v + (1 - beta2) * (g_t * g_t)
            v_hat = np.maximum(v_hat, v)
            eta_t = 1.0 / np.sqrt(v_hat)

            x[t+1] = x[t] - gamma * m * eta_t

            if t+1 in record_f:
                f[counter, trial] = loss.func(x[t+1], full_batch)
                counter += 1

    name = r'AdaSAM, $\gamma='+'{:.3f}'.format(gamma)+r'$, $\rho='+'{:.3f}'.format(rho)+r'$, $\beta_1='+'{:.3f}'.format(beta1)+r'$, $\beta_2='+'{:.3f}'.format(beta2)+r'$'
    name = 'AdaSAM'
    return name, f



###############
# LightSAM-I  #
###############
def LightSAM_I(loss, trials, record_f, x0, eta, rho, eps=1e-8, bs=1):
    # LightSAM-I (AdaGrad-Norm)
    # s_t = \nabla f(x_t, \xi_t)
    # u_t = u_{t-1} + ||s_t||^2
    # w_t = x_t + \rho * s_t / sqrt(u_t)
    # g_t = \nabla f(w_t, \xi_t)
    # v_t = v_{t-1} + ||g_t||^2
    # x_{t+1} = x_t - \eta * g_t / sqrt(v_t)

    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    T = record_f[-1]

    for trial in range(trials):
        x = [x0 for i in range(T+1)]
        u = eps**2
        v = eps**2
        counter = 1

        for t in range(T):
            i_t = np.random.choice(a=range(loss.n), size=bs)
            s_t = loss.grad(x[t], i_t)

            u = u + np.linalg.norm(s_t)**2
            w_t = x[t] + rho * s_t / np.sqrt(u)
            g_t = loss.grad(w_t, i_t)

            v = v + np.linalg.norm(g_t)**2
            x[t+1] = x[t] - eta * g_t / np.sqrt(v)

            if t+1 in record_f:
                f[counter, trial] = loss.func(x[t+1], full_batch)
                counter += 1

    name = r'LightSAM-I (AdaGrad-Norm), $\eta='+'{:.3f}'.format(eta)+r'$, $\rho='+'{:.3f}'.format(rho)+r'$'
    name = 'LightSAM-I'
    return name, f



###############
# LightSAM-II #
###############
def LightSAM_II(loss, trials, record_f, x0, eta, rho, eps=1e-8, bs=1):
    # LightSAM-II (AdaGrad)
    # s_t = \nabla f(x_t, \xi_t)
    # u_t = u_{t-1} + s_t \odot s_t
    # w_t = x_t + \rho * (1/sqrt(u_t)) \odot s_t
    # g_t = \nabla f(w_t, \xi_t)
    # v_t = v_{t-1} + g_t \odot g_t
    # x_{t+1} = x_t - \eta * (1/sqrt(v_t)) \odot g_t

    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    T = record_f[-1]

    for trial in range(trials):
        x = [x0 for i in range(T+1)]
        u = np.full_like(x0, eps**2, dtype=float)
        v = np.full_like(x0, eps**2, dtype=float)
        counter = 1

        for t in range(T):
            i_t = np.random.choice(a=range(loss.n), size=bs)
            s_t = loss.grad(x[t], i_t)

            u = u + s_t * s_t
            w_t = x[t] + rho * s_t / np.sqrt(u)
            g_t = loss.grad(w_t, i_t)

            v = v + g_t * g_t
            x[t+1] = x[t] - eta * g_t / np.sqrt(v)

            if t+1 in record_f:
                f[counter, trial] = loss.func(x[t+1], full_batch)
                counter += 1

    name = r'LightSAM-II (AdaGrad), $\eta='+'{:.3f}'.format(eta)+r'$, $\rho='+'{:.3f}'.format(rho)+r'$'
    name = 'LightSAM-II (AdaGrad)'
    return name, f



################
# LightSAM-III #
################
def LightSAM_III(loss, trials, record_f, x0, eta, rho, beta1, beta2, eps=1e-8, bs=1):
    # LightSAM-III (Adam)
    # s_t = \nabla f(x_t, \xi_t)
    # r_t = \beta_1 * r_{t-1} + (1-\beta_1) * s_t
    # u_t = \beta_2 * u_{t-1} + (1-\beta_2) * s_t \odot s_t
    # w_t = x_t + \rho * (1/sqrt(u_t)) \odot r_t
    # g_t = \nabla f(w_t, \xi_t)
    # m_t = \beta_1 * m_{t-1} + (1-\beta_1) * g_t
    # v_t = \beta_2 * v_{t-1} + (1-\beta_2) * g_t \odot g_t
    # x_{t+1} = x_t - \eta * (1/sqrt(v_t)) \odot m_t

    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    T = record_f[-1]

    for trial in range(trials):
        x = [x0 for i in range(T+1)]
        r = np.zeros_like(x0, dtype=float)
        u = np.full_like(x0, eps**2, dtype=float)
        m = np.zeros_like(x0, dtype=float)
        v = np.full_like(x0, eps**2, dtype=float)
        counter = 1

        for t in range(T):
            i_t = np.random.choice(a=range(loss.n), size=bs)
            s_t = loss.grad(x[t], i_t)

            r = beta1 * r + (1 - beta1) * s_t
            u = beta2 * u + (1 - beta2) * (s_t * s_t)
            w_t = x[t] + rho * r / np.sqrt(u)
            g_t = loss.grad(w_t, i_t)

            m = beta1 * m + (1 - beta1) * g_t
            v = beta2 * v + (1 - beta2) * (g_t * g_t)
            x[t+1] = x[t] - eta * m / np.sqrt(v)

            if t+1 in record_f:
                f[counter, trial] = loss.func(x[t+1], full_batch)
                counter += 1

    name = r'LightSAM-III (Adam), $\eta='+'{:.3f}'.format(eta)+r'$, $\rho='+'{:.3f}'.format(rho)+r'$, $\beta_1='+'{:.3f}'.format(beta1)+r'$, $\beta_2='+'{:.3f}'.format(beta2)+r'$'
    name = 'LightSAM-III (Adam)'
    return name, f



##########
# SA-SAM #
##########
def SA_SAM(loss, trials, record_f, x0, eta0, bs=1):
    # SA-SAM: Smoothness-Adaptive Sharpness-Aware Minimization
    # Initialization: w_0, \eta_0 > 0, \theta_0 = +\infty
    # w_1 = w_0 - \eta_0 * \nabla L(w_0 + \rho_0 * \nabla L(w_0;\xi_0)/||\nabla L(w_0;\xi_0)||; \xi_0)
    # For t = 1, ...:
    #   \eta_t = min{ ||w_t - w_{t-1}|| / (2 * ||\nabla L(w_t;\xi_t) - \nabla L(w_{t-1};\xi_t)||),
    #                 sqrt(1+\theta_{t-1}) * \eta_{t-1} }
    #   \rho_t = sqrt(\eta_t)
    #   w_{t+1} = w_t - \eta_t * \nabla L(w_t + \rho_t * \nabla L(w_t;\xi_t)/||\nabla L(w_t;\xi_t)||; \xi_t)
    #   \theta_t = \eta_t / \eta_{t-1}

    full_batch = np.arange(loss.n)
    f = np.zeros((len(record_f), trials))
    f[0, :] = loss.func(x0, full_batch)
    T = record_f[-1]

    for trial in range(trials):
        w = [x0 for i in range(T+1)]
        counter = 1

        # Initial step t=0
        i_0 = np.random.choice(a=range(loss.n), size=bs)
        g0 = loss.grad(w[0], i_0)
        rho_0 = np.sqrt(eta0)
        g0_sam = loss.grad(w[0] + rho_0 * g0 / np.linalg.norm(g0), i_0)
        w[1] = w[0] - eta0 * g0_sam

        if 1 in record_f:
            f[counter, trial] = loss.func(w[1], full_batch)
            counter += 1

        eta_prev = eta0
        # theta_0 = +inf, so sqrt(1+theta_0)*eta_0 = inf, no constraint from this on eta_1
        theta_prev = np.inf

        # Store previous gradient at w_t with batch i_t
        # Need grad at w_{t-1} with batch i_t for the difference - need to recompute
        # Following the formula literally: \nabla L(w_t; \xi_t) - \nabla L(w_{t-1}; \xi_t)
        # both with the same batch \xi_t

        for t in range(1, T):
            i_t = np.random.choice(a=range(loss.n), size=bs)
            g_t = loss.grad(w[t], i_t)
            g_tm1 = loss.grad(w[t-1], i_t)

            grad_diff_norm = np.linalg.norm(g_t - g_tm1)
            step_diff_norm = np.linalg.norm(w[t] - w[t-1])

            if grad_diff_norm > 0:
                cand1 = step_diff_norm / (2 * grad_diff_norm)
            else:
                cand1 = np.inf
            cand2 = np.sqrt(1 + theta_prev) * eta_prev
            eta_t = min(cand1, cand2)

            rho_t = np.sqrt(eta_t)
            g_t_sam = loss.grad(w[t] + rho_t * g_t / np.linalg.norm(g_t), i_t)
            w[t+1] = w[t] - eta_t * g_t_sam

            theta_prev = eta_t / eta_prev
            eta_prev = eta_t

            if t+1 in record_f:
                f[counter, trial] = loss.func(w[t+1], full_batch)
                counter += 1

    name = r'SA-SAM, $\eta_0='+'{:.3f}'.format(eta0)+r'$'
    name = 'SA-SAM'
    return name, f
