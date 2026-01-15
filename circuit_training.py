# src/utils/circuit_training.py

import os
import numpy as np
import torch
from tqdm import tqdm
from torch import optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models.ansatzes_shan2 import PQCAutoencoder
from src.utils.schedule import make_schedule, get_default_device
from src.utils.loss import infidelity_loss, LossHistory
from src.utils.training_functions import assemble_input, assemble_mu_tilde
from src.utils.plot_functions import show_mnist_alphas, log_generated_samples
from src.data.load_data import load_mnist

def training(path, config, data_length):
    """
    Qiskit Quantum Diffusion Model Training Loop
    Args:
        path: save path
        config: trainer.py dictionary
        data_length: size dataset
    """
    
    # 1. Device Setup
    device = get_default_device()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # 2. Unpack Configuration (Dictionary)
    num_qubits = config['num_qubits']
    trash_qubits = config['trash_qubits']
    model_type = config['model_type']
    pqc_layers = config['pqc_layers']
    activation = config['activation']
    return_trash_score = config['return_trash_score']
    device = config['device']
    layers = config['layers']

    T = config['T']
    beta0 = config['beta_0']
    betaT = config['beta_T']
    schedule = config['schedule']
    schedule_exponent = config['schedule_exponent']
     
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    pqc_lr = config['PQC_LR']
    wd_pqc = config['wd_PQC']
    init_variance = config['init_variance']
    
    desired_digits = config['digits']
    checkpoint = config['checkpoint']
    
    scheduler_patience = config['scheduler_patience']
    scheduler_gamma = config['scheduler_gamma']

    # 3. Training Setup
    betas, alphas_bar = make_schedule(beta0, betaT, T, schedule, schedule_exponent, device)
    best_loss = float('inf')
    sample_log_interval = 10  # interval of save image

    # Directories
    tensorboard_dir = os.path.join(path, 'TensorBoard')
    params_dir = os.path.join(path, 'Params')
    logs_dir = os.path.join(path, 'Logs')
    
    os.makedirs(path, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # 4. TensorBoard Logging
    writer = SummaryWriter(tensorboard_dir)
    hparam_dict = {k: (str(v) if isinstance(v, list) else v) for k, v in config.items()}
    writer.add_hparams(hparam_dict, {'hparam/best_loss': best_loss})

    # 5. Load Data
    dataset = load_mnist(desired_digits, data_length)
    show_mnist_alphas(dataset, alphas_bar, writer, device, height=16, width=16)

    # 6. Initialize Qiskit AutoEncoder PQCLayer
    print(f"Initializing Qiskit QuantumUNet on {device}...")
    print('layers:', layers)
    circuit =PQCAutoencoder(
        data_qubits=num_qubits,       # Input Dimension
        trash_qubits=1,
        layers=layers,           # PQC Reps
        model_type="bottleneck", # or "reverse_bottleneck"
        entanglement = "linear", # or "ring",
        useactivation = activation, 
        device=device,
    ).to(device)

    # 7. Optimizer
    optimizer = Adam(circuit.parameters(), lr=pqc_lr, weight_decay=wd_pqc)

    # 8. Load Checkpoint (If exists)
    if checkpoint is not None:
        param_path = checkpoint if os.path.isabs(checkpoint) else os.path.join(os.getcwd(), checkpoint)
        print(f"Loading checkpoint from: {param_path}")
        circuit.load_current_params(param_path)

    # 9. DataLoader
    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        num_workers=0
    )
    # 10. Scheduler & Loss Monitor
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_patience, gamma=scheduler_gamma)
    loss_monitor = LossHistory(logs_dir, len(data_loader))
    
    # 11. Training Loop
    print("Starting training...")
    for epoch in range(1, num_epochs + 1):

        epoch_losses = []
        epoch_progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch}/{num_epochs}")

        for _, image_batch in epoch_progress_bar:
            optimizer.zero_grad()

            # --- Data Preparation ---
            
            # shape: (BS*T)
            t = torch.tensor(range(1, T+1), dtype=torch.long).repeat_interleave(batch_size).to(device)
            print(t.shape)

            # shape: (BS*T, 2^num_qubits)
            image_batch = image_batch.to(device).repeat(T, 1).to(device)
            print(image_batch.shape)

            # Add Noise (Forward Diffusion)
            input_batch = assemble_input(image_batch, t, alphas_bar).to(torch.complex64)
            mu_tilde_t  = assemble_mu_tilde(image_batch, input_batch, t, alphas_bar, betas).to(torch.complex64)
            print(input_batch.shape, mu_tilde_t.shape)


            # Reshape for Model: (T, BS, Dim)
            input_batch = input_batch.view(T, batch_size, -1)
            mu_tilde_t  = mu_tilde_t.view(T, batch_size, -1)

            # Normalize Quantum States
            eps = 1e-8
            input_batch = input_batch / (torch.norm(input_batch.abs(), p=2, dim=-1, keepdim=True) + eps)
            mu_tilde_t  = mu_tilde_t / (torch.norm(mu_tilde_t.abs(), p=2, dim=-1, keepdim=True) + eps)
            print(input_batch.shape, mu_tilde_t.shape)

            # --- Forward Pass (Qiskit Circuit) ---
            predicted_mu_t = circuit(input_batch)

            # --- Loss Calculation ---
            losses = infidelity_loss(predicted_mu_t, mu_tilde_t)

            # Logging & Optimization
            loss_monitor.log_losses(losses, writer)
            loss = torch.mean(losses)
            loss.backward()
            optimizer.step()

            # Update Progress Bar
            epoch_progress_bar.set_postfix({'Loss': f"{loss.item():.5f}"})
            epoch_losses.append(loss.detach().item())

        # --- End of Epoch Operations ---
        writer.add_scalar('Epoch', epoch, epoch)
        scheduler.step()

        # Log LR
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Learning Rate', current_lr, epoch)

        # Save Parameters (Current)
        circuit.save_params(params_dir, epoch=epoch, best=False)     # Epoch
        circuit.save_params(params_dir, epoch=None, best=False)      # (current_model.pt)

        # Generate Samples (Visualization)
        if epoch % sample_log_interval == 0 or epoch == num_epochs:
            log_generated_samples(
                params_dir, epoch, T, num_qubits, writer,
                init_variance=init_variance, 
                betas=betas, 
                pqc_layers=pqc_layers,
                activation=activation, 
                bottleneck_qubits=bottleneck_qubits,
                num_samples=16
            )

        # Check Best Loss
        epoch_mean_loss = float(np.mean(epoch_losses))
        writer.add_scalar('Loss/epoch_mean', epoch_mean_loss, epoch)

        if epoch_mean_loss < best_loss:
            best_loss = epoch_mean_loss
            writer.add_scalar('Best Total Loss', best_loss, epoch)
            circuit.save_params(params_dir, best=True)
            print(f" -> New Best Loss: {best_loss:.5f}")

    writer.close()
    print("Training Complete.")
