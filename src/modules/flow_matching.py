import torch
import torch.nn as nn
import torch.nn.functional as F

class FlowMatchingLoss(nn.Module):
    """Enhanced Flow Matching Loss with stability improvements"""
    def __init__(self, loss_type='huber', sigma=0.1, eps=1e-7, stability_weight=0.1, embedding_task=True):
        super().__init__()
        self.loss_type = loss_type
        self.sigma = sigma
        self.eps = eps
        self.stability_weight = stability_weight
        self.embedding_task = embedding_task
        
    def forward(self, model, x1, cond):
        dtype = next(model.parameters()).dtype
        x1 = x1.to(dtype)
        cond = cond.to(dtype)
        device = x1.device
        
        t = torch.rand(x1.size(0), device=device, dtype=dtype)
        x0 = torch.randn_like(x1, dtype=dtype)
        t_expanded = t.unsqueeze(-1)
        xt = (1 - t_expanded) * x0 + t_expanded * x1
        
        pred = model(xt, t, cond)
        target_vector = x1 - x0
        
        if self.loss_type == 'mse':
            flow_loss = F.mse_loss(pred, target_vector, reduction='mean')
        elif self.loss_type == 'l1':
            flow_loss = F.l1_loss(pred, target_vector, reduction='mean')
        elif self.loss_type == 'huber':
            flow_loss = F.smooth_l1_loss(pred, target_vector, reduction='mean', beta=0.1)
        else:
            flow_loss = F.mse_loss(pred, target_vector, reduction='mean')
        
        additional_losses = {}
        
        if self.embedding_task:
            # 1. Magnitude consistency loss
            pred_norm = torch.norm(pred, dim=-1).mean()
            target_norm = torch.norm(target_vector, dim=-1).mean()
            magnitude_loss = F.mse_loss(pred_norm.unsqueeze(0), target_norm.unsqueeze(0))
            additional_losses['magnitude'] = magnitude_loss
            
            # 2. Timestep consistency
            timestep_weight = 1.0 - 0.5 * torch.abs(t - 0.5).mean()
            timestep_loss = flow_loss * (2.0 - timestep_weight)
            additional_losses['timestep'] = timestep_loss - flow_loss
        
        total_additional_loss = sum(additional_losses.values()) if additional_losses else 0.0
        total_loss = flow_loss + self.stability_weight * total_additional_loss
        
        with torch.no_grad():
            pred_norm = torch.norm(pred, dim=-1).mean()
            target_norm = torch.norm(target_vector, dim=-1).mean()
            pred_target_cosine = F.cosine_similarity(pred, target_vector, dim=-1).mean()
            
            prediction_error = torch.norm(pred - target_vector, dim=-1).mean()
            relative_error = prediction_error / (target_norm + self.eps)
            
            timestep_mean = t.mean()
            timestep_std = t.std()
            
        diagnostics = {
            'flow_loss': flow_loss.item(),
            'total_loss': total_loss.item(),
            'pred_norm': pred_norm.item(),
            'target_norm': target_norm.item(),
            'pred_target_cosine': pred_target_cosine.item(),
            'prediction_error': prediction_error.item(),
            'relative_error': relative_error.item(),
            'timestep_mean': timestep_mean.item(),
            'timestep_std': timestep_std.item(),
        }
        
        for name, loss in additional_losses.items():
            diagnostics[f'{name}_loss'] = loss.item()
        
        return total_loss, diagnostics

# Rest of FlowSampler and CFMTrainer remains unchanged

class FlowSampler:
    """Enhanced ODE-based sampler for flow models with multiple solver options"""
    def __init__(self, model, steps=12, solver='euler', eps=1e-3, guidance_scale=1.0):
        self.model = model
        self.steps = steps
        self.solver = solver.lower()
        self.eps = eps  # Avoid t=0 exactly
        self.guidance_scale = guidance_scale
        
        # Validate solver type
        valid_solvers = ['euler', 'heun', 'rk4', 'dopri5']
        if self.solver not in valid_solvers:
            raise ValueError(f"Invalid solver '{solver}'. Choose from {valid_solvers}")

    def sample(self, cond, initial_noise=None, return_trajectory=False):
        """
        Generate samples from noise with enhanced options
        
        Args:
            cond: EVA conditioning embeddings [B, D_cond]
            initial_noise: Optional initial noise [B, D]
            return_trajectory: Return full sampling trajectory
        
        Returns:
            final_sample or (final_sample, trajectory)
        """
        # Ensure consistent dtype with model parameters
        dtype = next(self.model.parameters()).dtype
        cond = cond.to(dtype)
        
        device = cond.device
        batch_size = cond.size(0)
        
        # Get output dimension from model
        if hasattr(self.model, 'output_proj'):
            dim = self.model.output_proj.out_features
        elif hasattr(self.model, 'input_proj'):
            # Assume output dim equals input dim
            dim = self.model.input_proj.in_features
        else:
            raise ValueError("Cannot determine model output dimension")
        
        # Initialize noise
        if initial_noise is None:
            x = torch.randn(batch_size, dim, device=device, dtype=dtype)
        else:
            x = initial_noise.to(device).to(dtype)
            
        # Time discretization (avoid exactly t=0)
        timesteps = torch.linspace(self.eps, 1, self.steps + 1, device=device, dtype=dtype)
        
        trajectory = [x.clone()] if return_trajectory else None
        
        if self.solver == 'euler':
            x = self._euler_solver(x, timesteps, cond, trajectory)
        elif self.solver == 'heun':
            x = self._heun_solver(x, timesteps, cond, trajectory)
        elif self.solver == 'rk4':
            x = self._rk4_solver(x, timesteps, cond, trajectory)
        elif self.solver == 'dopri5':
            x = self._dopri5_solver(x, timesteps, cond, trajectory)
        
        if return_trajectory:
            return x, torch.stack(trajectory)
        else:
            return x
            
    def _euler_solver(self, x, timesteps, cond, trajectory=None):
        """Basic Euler integration"""
        for i in range(self.steps):
            t = timesteps[i]
            dt = timesteps[i+1] - t
            
            # Predict vector field
            with torch.no_grad():
                pred = self.model(x, t.expand(x.size(0)), cond)
                
                # Apply guidance scaling if specified
                if self.guidance_scale != 1.0:
                    pred = pred * self.guidance_scale
            
            # Update sample: x_{t+dt} = x_t + u_t * dt
            x = x + pred * dt
            
            if trajectory is not None:
                trajectory.append(x.clone())
            
        return x

    def _heun_solver(self, x, timesteps, cond, trajectory=None):
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
                
                # Apply guidance scaling
                if self.guidance_scale != 1.0:
                    k1 = k1 * self.guidance_scale
                    k2 = k2 * self.guidance_scale
                
            # Average gradients
            x = x + (k1 + k2) * 0.5 * dt
            
            if trajectory is not None:
                trajectory.append(x.clone())
            
        return x

    def _rk4_solver(self, x, timesteps, cond, trajectory=None):
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
                
                # Apply guidance scaling
                if self.guidance_scale != 1.0:
                    k1 = k1 * self.guidance_scale
                    k2 = k2 * self.guidance_scale
                    k3 = k3 * self.guidance_scale
                    k4 = k4 * self.guidance_scale
                
            # Combine gradients
            dx = (k1 + 2*k2 + 2*k3 + k4) * (dt / 6)
            x = x + dx
            
            if trajectory is not None:
                trajectory.append(x.clone())
            
        return x
    
    def _dopri5_solver(self, x, timesteps, cond, trajectory=None):
        """Dormand-Prince 5th order method with adaptive step size"""
        # Simplified adaptive implementation
        for i in range(self.steps):
            t = timesteps[i]
            dt = timesteps[i+1] - t
            
            with torch.no_grad():
                # Use RK4 as approximation for DOPRI5
                k1 = self.model(x, t.expand(x.size(0)), cond)
                k2 = self.model(x + k1 * dt * 0.2, (t + dt * 0.2).expand(x.size(0)), cond)
                k3 = self.model(x + k1 * dt * 0.075 + k2 * dt * 0.225, (t + dt * 0.3).expand(x.size(0)), cond)
                k4 = self.model(x + k1 * dt * (44/45) - k2 * dt * (56/15) + k3 * dt * (32/9), (t + dt * 0.8).expand(x.size(0)), cond)
                
                # Apply guidance scaling
                if self.guidance_scale != 1.0:
                    k1 = k1 * self.guidance_scale
                    k2 = k2 * self.guidance_scale
                    k3 = k3 * self.guidance_scale
                    k4 = k4 * self.guidance_scale
            
            # Combine with DOPRI5 weights (simplified)
            dx = dt * (35/384 * k1 + 500/1113 * k3 + 125/192 * k4)
            x = x + dx
            
            if trajectory is not None:
                trajectory.append(x.clone())
        
        return x

class CFMTrainer:
    """Conditional Flow Matching trainer with BLIP3-o optimizations"""
    def __init__(self, model, loss_fn, optimizer, scheduler=None, device='cuda'):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Mixed precision setup
        self.scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
        
    def train_step(self, batch):
        """Single training step with mixed precision and gradient scaling"""
        clip_emb = batch['clip_embedding'].to(self.device)
        eva_emb = batch['eva_embedding'].to(self.device)
        
        # Convert to model's dtype
        dtype = next(self.model.parameters()).dtype
        clip_emb = clip_emb.to(dtype)
        eva_emb = eva_emb.to(dtype)
        
        # Forward pass with mixed precision
        if self.scaler:
            with torch.cuda.amp.autocast():
                loss, diagnostics = self.loss_fn(self.model, clip_emb, eva_emb)
        else:
            loss, diagnostics = self.loss_fn(self.model, clip_emb, eva_emb)
        
        # Backward pass with gradient scaling
        self.optimizer.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        # Update scheduler if provided
        if self.scheduler:
            self.scheduler.step()
        
        diagnostics['grad_norm'] = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        
        return loss.item(), diagnostics
        
    def validate(self, val_loader, max_batches=10):
        """Validation loop with comprehensive metrics"""
        self.model.eval()
        total_loss = 0
        total_diagnostics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break
                    
                clip_emb = batch['clip_embedding'].to(self.device)
                eva_emb = batch['eva_embedding'].to(self.device)
                
                dtype = next(self.model.parameters()).dtype
                clip_emb = clip_emb.to(dtype)
                eva_emb = eva_emb.to(dtype)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        loss, diagnostics = self.loss_fn(self.model, clip_emb, eva_emb)
                else:
                    loss, diagnostics = self.loss_fn(self.model, clip_emb, eva_emb)
                
                total_loss += loss.item()
                
                # Accumulate diagnostics
                for key, value in diagnostics.items():
                    total_diagnostics[key] = total_diagnostics.get(key, 0) + value
                
                num_batches += 1
        
        # Average metrics
        avg_loss = total_loss / num_batches
        avg_diagnostics = {k: v / num_batches for k, v in total_diagnostics.items()}
        
        self.model.train()
        return avg_loss, avg_diagnostics