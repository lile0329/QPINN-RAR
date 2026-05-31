import torch


def navier_stokes_2D_operator(model, t, x, y, min_x=0, max_x=1):
    """
    Operator to compute residuals for the 2D Navier-Stokes equation
    """

    mu = 0.00345
    DENSITY = 1056.0

    t.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True

    uvp = model(torch.concatenate((t, x, y), 1))

    u = uvp[:, 0:1]
    v = uvp[:, 1:2]
    p = uvp[:, 2:3]

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]

    v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

    continuity = u_x + v_y
    f_u = u_t + (u * u_x + v * u_y) + 1.0 / DENSITY * p_x - mu * (u_xx + u_yy)
    f_v = v_t + (u * v_x + v * v_y) + 1.0 / DENSITY * p_y - mu * (v_xx + v_yy)

    return [continuity, f_u, f_v]


def klein_gordon_operator(fluid_model, t, x, x_min=0.0, x_max=1.0):
    """
    Operator to compute residuals for the 1D Klein-Gordon equation
    """

    alpha = -1.0
    beta = 0.0
    gamma = 1.0
    k = 3

    t.requires_grad = True
    x.requires_grad = True

    u = fluid_model(torch.concatenate((t, x), 1))

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    residual = u_tt + alpha * u_xx + beta * u + gamma * u**k
    return u, residual


def wave_operator(model, t, x, sigma_t=1.0, sigma_x=1.0):
    """
    Operator to compute residuals for the 1D wave equation
    """
    c = 2
    t.requires_grad = True
    x.requires_grad = True

    u = model(torch.concatenate((t, x), 1))

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    residual = u_tt - c**2 * u_xx
    return u, residual


def diffusion_operator(
    model, t, x, y, sigma_t=1.0, sigma_x=1.0, sigma_y=1.0, D=0.01, v_x=1.0, v_y=1.0
):
    """
    Operator to compute residuals for the 2D convection-diffusion equation
    """

    t.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True

    # forward pass through the model
    u = model(torch.cat((t, x, y), 1))

    # compute derivatives
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0] / sigma_t
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0] / sigma_x
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0] / sigma_y

    u_xx = (
        torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        / sigma_x
    )
    u_yy = (
        torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
        / sigma_y
    )

    # convection-diffusion equation residual
    residual = u_t + v_x * u_x + v_y * u_y - D * (u_xx + u_yy)

    return u, residual


def burgers_operator(model, t, x, nu=None):
    """
    Operator to compute residuals for the 1D Burgers' equation.
    u_t + u * u_x = nu * u_xx
    """
    if nu is None:
        nu = 1.0 / (100.0 * torch.pi)

    # Ensure t and x are leaf tensors that require grad (avoid in-place on non-leaf)
    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)

    u = model(torch.cat((t, x), 1))

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    residual = u_t + u * u_x - nu * u_xx

    return u, residual


def diffusion_1d_operator(model, t, x, nu=None):
    """
    Operator to compute residuals for the 1D diffusion equation:
    u_t - nu * u_xx = 0
    """
    if nu is None:
        nu = 1.0 / (100.0 * torch.pi)

    # ensure grads can be computed
    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)

    u = model(torch.cat((t, x), 1))

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    residual = u_t - nu * u_xx

    return u, residual


def heat_3d_operator(model, t, x, y, z, alpha=0.1):
    """
    Operator to compute residuals for the 3D heat equation:
        u_t = alpha (u_xx + u_yy + u_zz)
    i.e.
        u_t - alpha (u_xx + u_yy + u_zz) = 0

    Args:
        model: neural network taking (t, x, y, z) as inputs
        t, x, y, z: coordinate tensors shaped (N, 1)
        alpha: thermal diffusivity

    Returns:
        u, residual
    """
    # ensure leaf tensors with gradients enabled
    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)

    u = model(torch.cat((t, x, y, z), dim=1))

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z, torch.ones_like(u), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z, z, torch.ones_like(u_z), create_graph=True)[0]

    residual = u_t - alpha * (u_xx + u_yy + u_zz)
    return u, residual


def allen_cahn_operator(model, t, x, epsilon=0.001, gamma=5.0):
    """
    Operator for the 1D Allen–Cahn equation:
        u_t = epsilon * u_xx + gamma * (u - u^3)
    which we rewrite as
        u_t - epsilon * u_xx - gamma * (u - u^3) = 0.
    """
    # ensure grads can be computed
    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)

    u = model(torch.cat((t, x), 1))

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    residual = u_t - epsilon * u_xx - gamma * (u - u**3)

    return u, residual


def kdv_operator(model, t, x, alpha=None):
    """
    Operator to compute residuals for the 1D Korteweg-de Vries (KdV) equation.
    u_t + u * u_x + alpha * u_xxx = 0
    
    Args:
        model: The neural network model
        t: Time coordinates
        x: Spatial coordinates
        alpha: Dispersion coefficient (default: 0.0025)
    
    Returns:
        u, residual: Solution and PDE residual
    """
    if alpha is None:
        alpha = 0.0025

    t.requires_grad = True
    x.requires_grad = True

    u = model(torch.cat((t, x), 1))

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]

    residual = u_t + u * u_x + alpha * u_xxx

    return u, residual


def helmholtz_operator(
    fluid_model,
    x1,
    x2,
):
    """
    Operator to compute residuals for the 2D helmholtz equation
    """

    LAMBDA = 1.0

    # Ensure x1 and x2 require grad for derivative computation
    # Clone to avoid in-place operations and ensure requires_grad=True
    # This is necessary for computing derivatives with respect to x1 and x2
    x1 = x1.clone().requires_grad_(True)
    x2 = x2.clone().requires_grad_(True)
    
    u = fluid_model(torch.concatenate((x1, x2), 1))

    # compute gradients with respect to x1 and x2
    u_x1 = torch.autograd.grad(
        u, x1, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    u_x2 = torch.autograd.grad(
        u, x2, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    u_xx1 = torch.autograd.grad(
        u_x1, x1, grad_outputs=torch.ones_like(u_x1), create_graph=True
    )[0]

    u_xx2 = torch.autograd.grad(
        u_x2, x2, grad_outputs=torch.ones_like(u_x2), create_graph=True
    )[0]

    residual = u_xx1 + u_xx2 + LAMBDA * u

    return [u, residual]


def poisson_2d_operator(model, x1, x2):
    """
    Operator to compute residuals for the 2D Poisson equation
        -Δu = f
    We return u and Δu (without the minus sign); the training code
    will compare this to the target forcing term -f.
    """
    # Ensure x1 and x2 require grad for derivative computation
    x1 = x1.clone().detach().requires_grad_(True)
    x2 = x2.clone().detach().requires_grad_(True)

    u = model(torch.cat((x1, x2), dim=1))

    u_x1 = torch.autograd.grad(
        u, x1, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x2 = torch.autograd.grad(
        u, x2, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    u_xx1 = torch.autograd.grad(
        u_x1, x1, grad_outputs=torch.ones_like(u_x1), create_graph=True
    )[0]
    u_xx2 = torch.autograd.grad(
        u_x2, x2, grad_outputs=torch.ones_like(u_x2), create_graph=True
    )[0]

    laplace_u = u_xx1 + u_xx2
    return [u, laplace_u]


def laplace_polar_operator(model, r, theta):
    """
    Operator to compute residuals for the Laplace equation in polar coordinates:
        r^2 u_rr + r u_r + u_{θθ} = 0

    Args:
        model: neural network taking (r, θ) as inputs
        r: radial coordinate tensor
        theta: angular coordinate tensor

    Returns:
        u, residual  where residual = r^2 u_rr + r u_r + u_{θθ}
    """
    # Ensure r and theta require grad for derivative computation
    r = r.clone().detach().requires_grad_(True)
    theta = theta.clone().detach().requires_grad_(True)

    u = model(torch.cat((r, theta), dim=1))

    u_r = torch.autograd.grad(
        u, r, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_theta = torch.autograd.grad(
        u, theta, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    u_rr = torch.autograd.grad(
        u_r, r, grad_outputs=torch.ones_like(u_r), create_graph=True
    )[0]
    u_thetatheta = torch.autograd.grad(
        u_theta, theta, grad_outputs=torch.ones_like(u_theta), create_graph=True
    )[0]

    residual = (r**2) * u_rr + r * u_r + u_thetatheta
    return u, residual


def beam_vibration_operator(model, t, x, L=1.0):
    """
    Operator to compute residuals for the 1D beam vibration equation (fourth-order PDE):
        ∂²u/∂t² + ∂⁴u/∂x⁴ = 0
    
    Args:
        model: The neural network model
        t: Time coordinates
        x: Spatial coordinates
        L: Length of the beam (default: 1.0)
    
    Returns:
        u, residual: Solution and PDE residual
    """
    # Ensure t and x are leaf tensors that require grad
    t = t.clone().detach().requires_grad_(True)
    x = x.clone().detach().requires_grad_(True)

    u = model(torch.cat((t, x), 1))

    # First-order derivatives
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

    # Second-order derivatives
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    # Third-order derivative
    u_xxx = torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0]

    # Fourth-order derivative
    u_xxxx = torch.autograd.grad(u_xxx, x, torch.ones_like(u_xxx), create_graph=True)[0]

    # PDE residual: u_tt + u_xxxx = 0
    residual = u_tt + u_xxxx

    return u, residual


