import torch
import wandb
import lightning.pytorch as pl
from tnp.utils.experiment_utils import initialize_evaluation
from tnp.utils.np_functions import ar_loss_fn
from tnp.data.tempopfn_generator import TempoPFNGenerator

def main():
    experiment = initialize_evaluation()

    lit_model = experiment.lit_model
    eval_name = experiment.misc.eval_name
    
    # Check for the correct test generator
    if hasattr(experiment.generators, 'test'):
        gen_test = experiment.generators.test
    elif hasattr(experiment.generators, 'inc_test'):
        gen_test = experiment.generators.inc_test
    else:
        raise ValueError("Could not find test generator in experiment.generators")

    lit_model.eval()

    if torch.cuda.is_available():
        lit_model.cuda()

    # Extract the underlying PyTorch model to use our custom loss function
    model = lit_model.model
    device = next(model.parameters()).device
    
    # Store number of parameters
    num_params = sum(p.numel() for p in lit_model.parameters())

    print("\n" + "=" * 50)
    print(f"Starting Autoregressive Target Evaluation: {eval_name}")
    print(f"Model Class: {type(model).__name__}")
    print("=" * 50)

    target_nlls = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(gen_test):
            print(batch_idx, end="\r")  # Progress indicator
            # Ensure batch tensors are on the correct device
            batch.xc = batch.xc.to(device)
            batch.yc = batch.yc.to(device)
            batch.xt = batch.xt.to(device)
            if hasattr(batch, 'yt') and batch.yt is not None:
                batch.yt = batch.yt.to(device)
            
            # ar_loss_fn returns the NLL over the target sequence
            nll = ar_loss_fn(model, batch)
            target_nlls.append(nll)

    # Calculate statistics identical to eval.py
    loglik_tensor = -torch.stack(target_nlls)  # Log-likelihood is negative NLL
    mean_loglik = loglik_tensor.mean().item()
    std_loglik = (loglik_tensor.std() / (len(loglik_tensor) ** 0.5)).item()
    
    # Calculate NLL statistics for logging
    nll_tensor = torch.stack(target_nlls)
    mean_nll = nll_tensor.mean().item()
    std_nll = (nll_tensor.std() / (len(nll_tensor) ** 0.5)).item()

    # --- Start of Augmented Print Statements ---
    print("\n" + "=" * 40)
    print(f"AR Evaluation Results: {eval_name}")
    print(f"Number of Parameters: {num_params}")
    print("-" * 40)
    print(f"Mean AR Target NLL: {mean_nll:.4f} ± {std_nll:.4f}")
    print(f"Mean AR Target LogLik: {mean_loglik:.4f} ± {std_loglik:.4f}")
    print("=" * 40 + "\n")
    # --- End of Augmented Print Statements ---

    # Log to Weights & Biases
    if experiment.misc.logging:
        wandb.run.summary["num_params"] = num_params
        # Logging AR NLL as requested
        wandb.run.summary[f"test/{eval_name}/ar_nll"] = mean_nll
        wandb.run.summary[f"test/{eval_name}/std_ar_nll"] = std_nll
        # Also logging loglik to maintain consistency with standard eval.py tracking
        wandb.run.summary[f"test/{eval_name}/ar_loglik"] = mean_loglik
        wandb.run.summary[f"test/{eval_name}/std_ar_loglik"] = std_loglik

if __name__ == "__main__":
    main()