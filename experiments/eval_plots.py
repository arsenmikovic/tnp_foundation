from plot import plot, plot_discrete

import wandb
import torch
from tnp.utils.experiment_utils import initialize_evaluation
from tnp.utils.np_functions import np_pred_fn, ar_pred_fn

def main():
    experiment = initialize_evaluation()

    model = experiment.model
    eval_name = experiment.misc.eval_name
    gen_test = experiment.generators.ar_test

    # --- START OF FIX: moving model to CUDA
    # 1. Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")

    # 2. Move model to device
    model = model.to(device)
    model.eval()
    # --- END OF FIX ---

    gen_test.batch_size = 1
    gen_test.num_batches = experiment.misc.num_plots
    batches = list(iter(gen_test))

    # --- START OF FIX ---
    # 3. Move batches to device manually
    # Since Batch is a dataclass, we must move the tensors explicitly.
    for batch in batches:
        batch.xc = batch.xc.to(device)
        batch.yc = batch.yc.to(device)
        batch.xt = batch.xt.to(device)
        batch.yt = batch.yt.to(device)
        # Move other fields if your model uses them (e.g. x, y for some generators)
        if hasattr(batch, 'x'): batch.x = batch.x.to(device)
        if hasattr(batch, 'y'): batch.y = batch.y.to(device)
    # --- END OF FIX ---
    
    eval_folder = getattr(experiment.misc, 'eval_folder', 'eval')
    eval_name = eval_name
    if experiment.misc.plot_ar_updates:
        plot_discrete(
            model=model,
            batches=batches,
            num_fig=min(experiment.misc.num_plots, len(batches)),
            savefig=experiment.misc.savefig,
            logging=experiment.misc.logging,
            pred_fn=ar_pred_fn,
            outfolder=eval_folder,
            show_nll=experiment.misc.show_nll,
            separate_targets=False
        )
    else:
        plot_discrete(
            model=model,
            batches=batches,
            num_fig=min(experiment.misc.num_plots, len(batches)),
            savefig=experiment.misc.savefig,
            logging=experiment.misc.logging,
            pred_fn=np_pred_fn,
            outfolder=eval_folder,
            show_nll=experiment.misc.show_nll,
            separate_targets=True,
        )


if __name__ == "__main__":
    main()
