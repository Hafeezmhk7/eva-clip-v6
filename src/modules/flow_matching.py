import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatchingLoss(nn.Module):
    """Implements conditional flow matching loss with stability improvements"""
    def __init__(self, sigma=0.1, eps=1e-7):
        super().__init__()
        self.sigma = sigma
        self.eps = eps  # Prevents division by zero
        
    def forward(self, model, x1, cond):
        """
        model: LuminaDiT instance
        x1: Target CLIP embeddings [B, D] (ground truth)
        cond: EVA conditioning embeddings [B, D_cond]
        """
        # Ensure consistent dtype with model parameters
        dtype = next(model.parameters()).dtype
        x1 = x1.to(dtype)
        cond = cond.to(dtype)
        
        device = x1.device
        
        # Sample random timestep - ensure proper broadcasting
        t = torch.rand(x1.size(0), device=device, dtype=dtype)  # [B]
        
        # Sample noise from standard normal
        x0 = torch.randn_like(x1, dtype=dtype)  # Source noise [B, D]
        
        # Compute interpolated point (xt)
        # Use unsqueeze for proper broadcasting: [B] -> [B, 1] for element-wise ops
        t_expanded = t.unsqueeze(-1)  # [B, 1]
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        
        # Predict vector field at (xt, t)
        pred = model(xt, t, cond)  # [B, D]
        
        # Compute target vector field: u_t = x1 - x0
        target_vector = x1 - x0
        
        # MSE loss with optional sigma weighting
        return F.mse_loss(pred, target_vector, reduction='mean')

class FlowSampler:
    """ODE-based sampler for flow models with multiple solver options"""
    def __init__(self, model, steps=12, solver='euler', eps=1e-3):
        self.model = model
        self.steps = steps
        self.solver = solver.lower()
        self.eps = eps  # Avoid t=0 exactly
        
        # Validate solver type
        valid_solvers = ['euler', 'heun', 'rk4']
        if self.solver not in valid_solvers:
            raise ValueError(f"Invalid solver '{solver}'. Choose from {valid_solvers}")

    def sample(self, cond, initial_noise=None):
        """
        Generate samples from noise
        cond: EVA conditioning embeddings [B, D_cond]
        """
        # Ensure consistent dtype with model parameters
        dtype = next(self.model.parameters()).dtype
        cond = cond.to(dtype)
        
        device = cond.device
        batch_size = cond.size(0)
        dim = self.model.output_proj.out_features
        
        # Initialize noise
        if initial_noise is None:
            x = torch.randn(batch_size, dim, device=device, dtype=dtype)
        else:
            x = initial_noise.to(device).to(dtype)
            
        # Time discretization (avoid exactly t=0)
        timesteps = torch.linspace(self.eps, 1, self.steps + 1, device=device, dtype=dtype)
        
        if self.solver == 'euler':
            return self._euler_solver(x, timesteps, cond)
        elif self.solver == 'heun':
            return self._heun_solver(x, timesteps, cond)
        elif self.solver == 'rk4':
            return self._rk4_solver(x, timesteps, cond)
            
    def _euler_solver(self, x, timesteps, cond):
        """Basic Euler integration"""
        for i in range(self.steps):
            t = timesteps[i]
            dt = timesteps[i+1] - t
            
            # Predict vector field
            with torch.no_grad():
                pred = self.model(x, t.expand(x.size(0)), cond)
            
            # Update sample: x_{t+dt} = x_t + u_t * dt
            x = x + pred * dt
            
        return x

    def _heun_solver(self, x, timesteps, cond):
        """Heun's method (2nd-order Runge-Kutta) for better accuracy"""
        for i in range(self.steps):
            t = timesteps[i]
            dt = timesteps[i+1] - t
            t_next = t + dt
            
            # First prediction (Euler step)
            with torch.no_grad():
                k1 = self.model(x, t.expand(x.size(0)), cond)
                x_temp = x + k1 * dt
                
                # Correction term
                k2 = self.model(x_temp, t_next.expand(x.size(0)), cond)
                
            # Average gradients
            x = x + (k1 + k2) * 0.5 * dt
            
        return x

    def _rk4_solver(self, x, timesteps, cond):
        """Classic 4th-order Runge-Kutta for highest accuracy"""
        for i in range(self.steps):
            t = timesteps[i]
            dt = timesteps[i+1] - t
            half_dt = 0.5 * dt
            t_half = t + half_dt
            t_next = t + dt
            
            with torch.no_grad():
                # Step 1
                k1 = self.model(x, t.expand(x.size(0)), cond)
                # Step 2
                k2 = self.model(x + k1 * half_dt, t_half.expand(x.size(0)), cond)
                # Step 3
                k3 = self.model(x + k2 * half_dt, t_half.expand(x.size(0)), cond)
                # Step 4
                k4 = self.model(x + k3 * dt, t_next.expand(x.size(0)), cond)
                
            # Combine gradients
            dx = (k1 + 2*k2 + 2*k3 + k4) * (dt / 6)
            x = x + dx
            
        return x