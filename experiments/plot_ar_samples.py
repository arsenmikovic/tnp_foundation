import os
import torch
import wandb
import matplotlib.pyplot as plt

from plot import plot
from tnp.utils.experiment_utils import initialize_evaluation
from tnp.utils.np_functions import sample_function_trajectory


def plot_discrete_trajectory(
    batches,
    sampled_yts,
    num_fig=5,
    name="plot",
    outfolder="fig",
):
    """
    Plots the ground truth sequence against autoregressively sampled trajectories.
    """
    for i in range(num_fig):
        batch = batches[i]
        batch_samples = sampled_yts[i]
        num_trajectories = len(batch_samples)
        
        # Extract and Flatten (assuming Batch Size = 1)
        xc = batch.xc[0, :, 0].cpu().numpy()
        yc = batch.yc[0, :, 0].cpu().numpy()
        xt = batch.xt[0, :, 0].cpu().numpy()
        yt_gt = batch.yt[0, :, 0].cpu().numpy()

        fig = plt.figure(figsize=(10, 6))
        
        # A. Context Points
        plt.scatter(xc, yc, color='black', label='Context', s=30, zorder=5)
        
        # B. Ground Truth Target Points
        plt.scatter(xt, yt_gt, color='gray', alpha=0.7, label='Ground Truth Target', s=30, zorder=4)
        
        # C. Autoregressive Trajectories
        # Sort by X to ensure the lines draw continuously left-to-right
        sort_idx = xt.argsort()
        
        # Adjust styling based on the number of trajectories to keep it looking nice
        line_alpha = 0.8 if num_trajectories == 1 else max(0.15, 1.0 / (num_trajectories ** 0.6))
        line_width = 2.0 if num_trajectories == 1 else 1.5
        
        for j, sampled_yt in enumerate(batch_samples):
            yt_pred = sampled_yt[0, :, 0].cpu().numpy()
            
            # Only label the first trajectory to avoid legend duplication
            label = 'AR Sampled Trajectory' if j == 0 else None
            
            plt.plot(xt[sort_idx], yt_pred[sort_idx], color='blue', alpha=line_alpha, label=label, linewidth=line_width, zorder=2)
            
            # Only add small markers on the line if there are very few trajectories
            if num_trajectories <= 3:
                plt.scatter(xt, yt_pred, color='blue', s=15, alpha=line_alpha, zorder=3)

        gen0 = batch.generator_name[0] if hasattr(batch, "generator_name") else "unknown"
        
        plt.title(f"Generator: {gen0} (AR Trajectories: {num_trajectories})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # --- LOGGING ---
        fname = f"{outfolder}/{name}/{i:03d}"
        if wandb.run is not None:
            wandb.log({fname: wandb.Image(fig)})
        else:
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            plt.savefig(f"{fname}.png")

        plt.close(fig)


def main():
    experiment = initialize_evaluation()

    model = experiment.model
    eval_name = experiment.misc.eval_name
    gen_test = experiment.generators.ar_test

    # 1. Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running AR evaluation plots on: {device}")

    # 2. Move model to device
    model = model.to(device)
    model.eval()

    gen_test.batch_size = 1
    gen_test.num_batches = getattr(experiment.misc, 'num_plots', 5)
    batches = list(iter(gen_test))

    num_trajectories = getattr(experiment.misc, 'num_trajectories', 1)
    sampled_yts = []

    # 3. Move batches to device and generate AR samples
    for batch in batches:
        batch.xc = batch.xc.to(device)
        batch.yc = batch.yc.to(device)
        batch.xt = batch.xt.to(device)
        batch.yt = batch.yt.to(device)
        
        if hasattr(batch, 'x'): batch.x = batch.x.to(device)
        if hasattr(batch, 'y'): batch.y = batch.y.to(device)

        batch_samples = []
        
        # Sample the trajectory multiple times
        with torch.no_grad():
            for _ in range(num_trajectories):
                # We can loop safely here because sample_function_trajectory 
                # handles the initialisation of the incremental caches internally.
                sampled_yt = sample_function_trajectory(model, batch)
                batch_samples.append(sampled_yt)
                
        sampled_yts.append(batch_samples)
    
    # 4. Extract config arguments
    ar_folder = getattr(experiment.misc, 'ar_folder', 'ar_plots')
    plot_type = getattr(experiment.misc, 'plot_type', 'standard')
    
    # 5. Call the correct plotting utility
    if plot_type == "discrete":
        plot_discrete_trajectory(
            batches=batches,
            sampled_yts=sampled_yts,
            num_fig=min(gen_test.num_batches, len(batches)),
            name=eval_name + "_ar_trajectory",
            outfolder=ar_folder,
        )
    else:
        # If standard plot, overwrite the ground-truth targets with our FIRST sample.
        # This is required because standard plot does not natively handle multiple target lists.
        for batch, batch_samples in zip(batches, sampled_yts):
            batch.yt = batch_samples[0]

        plot(
            model=model,
            batches=batches,
            num_fig=min(gen_test.num_batches, len(batches)),
            name=eval_name + "_ar_trajectory",
            savefig=getattr(experiment.misc, 'savefig', True),
            logging=getattr(experiment.misc, 'logging', False),
            pred_fn=experiment.misc.pred_fn,
            plot_gt=getattr(experiment.misc, 'plot_gt', True),
            x_range=getattr(experiment.misc, 'plot_x_range', None),
            plot_reversal=getattr(experiment.misc, 'plot_reversal', False),
            outfolder=ar_folder,
        )


if __name__ == "__main__":
    main()