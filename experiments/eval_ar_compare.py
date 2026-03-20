import sys
import argparse
import pandas as pd
import torch
import wandb

from tnp.utils.experiment_utils import initialize_evaluation
from tnp.utils.np_functions import ar_loss_fn
from tnp.data.tempopfn_generator import TempoPFNGenerator

def run_single_ar_evaluation(run_path, checkpoint, config_path, model_name):
    """
    Runs the AR evaluation logic for a single model by manipulating sys.argv
    to mimic a command line call.
    """
    print(f"\n{'='*20}\nAR Evaluating: {model_name}\nRun: {run_path}\nCheckpoint: {checkpoint}\n{'='*20}")

    # 1. Save the original sys.argv
    original_argv = sys.argv[:]

    # 2. Mock the command line arguments for initialize_evaluation
    sys.argv = [
        "eval_ar.py", 
        "--run_path", run_path, 
        "--config", config_path, 
        "--checkpoint", checkpoint
    ]

    try:
        # 3. Initialise the experiment
        experiment = initialize_evaluation()
        
        lit_model = experiment.lit_model
        
        if hasattr(experiment.generators, 'test'):
            gen_test = experiment.generators.test
        elif hasattr(experiment.generators, 'inc_test'):
            gen_test = experiment.generators.inc_test
        else:
            raise ValueError("Could not find test generator in experiment.generators")

        lit_model.eval()

        if torch.cuda.is_available():
            lit_model.cuda()

        # Extract the underlying PyTorch model
        model = lit_model.model
        device = next(model.parameters()).device
        num_params = sum(p.numel() for p in lit_model.parameters())

        target_nlls = []

        # 4. Run the AR Test Loop
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

        # 5. Extract Results
        loglik_tensor = -torch.stack(target_nlls)  # Log-likelihood is negative NLL
        mean_loglik = loglik_tensor.mean().item()
        std_loglik = (loglik_tensor.std() / (len(loglik_tensor) ** 0.5)).item()
        
        nll_tensor = torch.stack(target_nlls)
        mean_nll = nll_tensor.mean().item()
        std_nll = (nll_tensor.std() / (len(nll_tensor) ** 0.5)).item()

        # Print the immediate result for the current model
        print("\n" + "-" * 40)
        print(f"Results for {model_name}:")
        print(f"Mean AR Target NLL:    {mean_nll:.4f} ± {std_nll:.4f}")
        print(f"Mean AR Target LogLik: {mean_loglik:.4f} ± {std_loglik:.4f}")
        print("-" * 40 + "\n")

        # 6. Return Data Row
        return {
            "model_name": model_name,
            "run_path": run_path,
            "checkpoint": checkpoint,
            "num_params": num_params,
            "mean_ar_loglik": mean_loglik,
            "std_ar_loglik": std_loglik,
            "mean_ar_nll": mean_nll,
            "std_ar_nll": std_nll
        }

    except Exception as e:
        print(f"!! Error evaluating {model_name} ({run_path}): {e}")
        return {
            "model_name": model_name,
            "run_path": run_path,
            "checkpoint": checkpoint,
            "error": str(e)
        }

    finally:
        # 7. Cleanup
        sys.argv = original_argv
        if wandb.run is not None:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Batch Autoregressive Evaluation Script")
    
    parser.add_argument(
        "--model_names",
        nargs='+',
        required=True,
        help="List of friendly names for the models (must match order of run_paths)"
    )
    parser.add_argument(
        "--run_paths", 
        nargs='+', 
        required=True, 
        help="List of WandB run paths (e.g., entity/project/run_id)"
    )
    parser.add_argument(
        "--checkpoints", 
        nargs='+', 
        required=True, 
        help="List of checkpoint artifacts. Must match the order and length of --run_paths"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the generator config file"
    )
    parser.add_argument(
        "--output_csv", 
        type=str, 
        default="batch_ar_eval_results.csv", 
        help="Path to save the output CSV"
    )

    args = parser.parse_args()

    # Input Validation
    if not (len(args.run_paths) == len(args.checkpoints) == len(args.model_names)):
        print(f"Error: Mismatch in argument lengths.")
        print(f"Model Names: {len(args.model_names)}")
        print(f"Run Paths:   {len(args.run_paths)}")
        print(f"Checkpoints: {len(args.checkpoints)}")
        sys.exit(1)

    results = []

    for run_path, checkpoint, model_name in zip(args.run_paths, args.checkpoints, args.model_names):
        result_data = run_single_ar_evaluation(run_path, checkpoint, args.config, model_name)
        results.append(result_data)

    df = pd.DataFrame(results)
    
    # Reorder columns for readability
    cols = ["model_name", "run_path", "checkpoint", "num_params", "mean_ar_loglik", "std_ar_loglik", "mean_ar_nll", "std_ar_nll"]
    
    if "error" in df.columns:
        cols.append("error")
    
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    print("\n" + "="*40)
    print("FINAL AR RESULTS FOR ALL MODELS")
    print("="*40)
    print(df.to_string())
    
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to: {args.output_csv}")

if __name__ == "__main__":
    main()