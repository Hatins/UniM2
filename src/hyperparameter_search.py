import argparse
import optuna
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import PercentilePruner
import subprocess
import re
from pathlib import Path
import yaml
from datetime import datetime
from tqdm import tqdm
import shutil
import os


def create_code_snapshot(work_dir, snapshot_dir, search_name):
    """Create a code snapshot so every trial runs the same source revision."""
    snapshot_path = Path(snapshot_dir) / search_name
    
    if snapshot_path.exists():
        shutil.rmtree(snapshot_path)
    
    snapshot_path.mkdir(parents=True)
    
    src_dir = Path(work_dir) / "src"
    snapshot_src = snapshot_path / "src"
    
    print(f"\nCreating code snapshot...")
    print(f"  Source: {src_dir}")
    print(f"  Snapshot: {snapshot_src}")
    
    shutil.copytree(src_dir, snapshot_src, 
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo', '.git*', 'wandb'))
    
    print(f"  Code snapshot created\n")
    return snapshot_path

def objective(trial: Trial, args, base_config, search_name, study, snapshot_dir):
    # Keep the categorical space fixed across all trials.
    dim_candidates = [70, 90, 128, 192, 256, 320, 384]
    
    dino_feat_type = base_config.get('dino_feat_type', 'ori')
    search_dim = base_config.get('search_dim', False)
    fixed_dim = base_config.get('dim', 70)
    
    search_lr = base_config.get('search_lr', False)
    fixed_lr = float(base_config.get('lr', 5e-4))
    
    modals = base_config.get('modals', ['rgb', 'depth'])
    
    PARAM_KEYS = ['pos_inter_weight', 'pos_intra_weight', 'neg_inter_weight',
                  'neg_inter_shift', 'pos_inter_shift', 'pos_intra_shift', 'weight']
    DEFAULTS   = [0.6313,            0.7939,             0.8754,
                  0.7358,            0.3906,             0.8975,            0.5]
    
    modal_baselines = {}
    for modal in modals:
        modal_baselines[modal] = {
            key: base_config.get(f'{modal}_{key}', default)
            for key, default in zip(PARAM_KEYS, DEFAULTS)
        }
        if f'{modal}_weight' not in base_config:
            modal_baselines[modal]['weight'] = 1.0 / len(modals)
    
    # DataLoader seed
    baseline_dataloader_seed = 18
    baseline_dim = base_config.get('dim', 70)
    baseline_lr  = float(base_config.get('lr', 5e-4))

    # Branch 1: search only the data-loader seed and keep all other parameters fixed.
    if args.search_only_seed and trial.number > 0:
        print(f"\n{'='*60}")
        print(f"Trial {trial.number} - Search Only Seed Mode")
        print(f"  All hyperparameters fixed at baseline values")
        print(f"{'='*60}\n")
        
        modal_params = {}
        for modal in modals:
            modal_params[modal] = {}
            for key in PARAM_KEYS:
                base_val = modal_baselines[modal][key]
                modal_params[modal][key] = trial.suggest_float(f'{modal}_{key}', base_val, base_val)
        
        dim = baseline_dim
        trial.set_user_attr('dim', baseline_dim)
        trial.set_user_attr('dim_is_baseline', True)
        lr = baseline_lr
        trial.set_user_attr('lr', baseline_lr)
        trial.set_user_attr('lr_is_baseline', True)
        
        all_seeds = list(range(0, 199))
        used_seeds = set()
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE and t.number != trial.number:
                seed_val = t.user_attrs.get('dataloader_seed_actual', t.params.get('dataloader_seed'))
                if seed_val is not None:
                    used_seeds.add(seed_val)
        suggested_seed = trial.suggest_int('dataloader_seed', 0, 20)
        remaining_seeds = [s for s in all_seeds if s not in used_seeds]
        if suggested_seed in used_seeds and remaining_seeds:
            dataloader_seed = remaining_seeds[0]
            trial.set_user_attr('dataloader_seed_actual', dataloader_seed)
            print(f"  [Dedup] Seed {suggested_seed} already used, switching to {dataloader_seed}")
        else:
            dataloader_seed = suggested_seed

    # Branch 2: run trial 0 with the baseline values from the config.
    elif trial.number == 0:
        print(f"\n{'='*60}")
        print(f"Trial 0 - Using baseline from config file:")
        for modal in modals:
            print(f"  {modal.capitalize()} Modality:")
            for key in PARAM_KEYS:
                print(f"    {modal}_{key}: {modal_baselines[modal][key]}")
        print(f"  DataLoader:")
        print(f"    dataloader_seed: {baseline_dataloader_seed}")
        print(f"  Other:")
        print(f"    dim: {baseline_dim}")
        print(f"    lr: {baseline_lr:.2e}")
        print(f"    dino_feat_type: {dino_feat_type}")
        print(f"    search_dim: {search_dim}")
        print(f"    search_lr: {search_lr}")
        print(f"{'='*60}\n")
        
        modal_params = {}
        for modal in modals:
            modal_params[modal] = {}
            for key in PARAM_KEYS:
                base_val = modal_baselines[modal][key]
                modal_params[modal][key] = trial.suggest_float(f'{modal}_{key}', base_val, base_val)
        
        dim = baseline_dim
        trial.set_user_attr('dim', baseline_dim)
        trial.set_user_attr('dim_is_baseline', True)
        lr = baseline_lr
        trial.set_user_attr('lr', baseline_lr)
        trial.set_user_attr('lr_is_baseline', True)
        
        if args.search_dataloader_seed:
            dataloader_seed = trial.suggest_int('dataloader_seed', baseline_dataloader_seed, baseline_dataloader_seed)
        else:
            dataloader_seed = baseline_dataloader_seed

    # Branch 3: normal hyperparameter search.
    else:
        modal_params = {}
        for modal in modals:
            modal_params[modal] = {}
            for key in PARAM_KEYS[:-1]:  # pos/neg weights and shifts
                modal_params[modal][key] = trial.suggest_float(f'{modal}_{key}', 0.0, 1.0)
            # Search each modal weight independently; training falls back to 1/n_modals.
            modal_params[modal]['weight'] = trial.suggest_float(f'{modal}_weight', 0.0, 1.0)
        
        if args.search_dataloader_seed:
            dataloader_seed = trial.suggest_int('dataloader_seed', 0, 20)
        else:
            dataloader_seed = baseline_dataloader_seed
        
        if search_dim:
            dim = trial.suggest_categorical('dim', dim_candidates)
        else:
            dim = fixed_dim
            trial.set_user_attr('dim_fixed', True)
        
        if search_lr:
            lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        else:
            lr = fixed_lr
            trial.set_user_attr('lr_fixed', True)
    
    # Print the sampled parameters for this trial.
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    for modal in modals:
        print(f"  {modal.capitalize()} Modality:")
        for key in PARAM_KEYS:
            print(f"    {modal}_{key}: {modal_params[modal][key]:.4f}")
    print(f"  DataLoader:")
    print(f"    dataloader_seed: {dataloader_seed}")
    print(f"  Other:")
    if search_dim:
        print(f"    dim: {dim} (searched)")
    else:
        print(f"    dim: {dim} (fixed)")
    if search_lr:
        print(f"    lr: {lr:.2e} (searched)")
    else:
        print(f"    lr: {lr:.2e} (fixed)")
    print(f"    dino_feat_type: {dino_feat_type}")
    print(f"{'='*60}\n")
    
    import os
    for modal in modals:
        modal_prefix = modal.upper()
        os.environ[f'OPTUNA_{modal_prefix}_POS_INTER_WEIGHT'] = str(modal_params[modal]['pos_inter_weight'])
        os.environ[f'OPTUNA_{modal_prefix}_POS_INTRA_WEIGHT'] = str(modal_params[modal]['pos_intra_weight'])
        os.environ[f'OPTUNA_{modal_prefix}_NEG_INTER_WEIGHT'] = str(modal_params[modal]['neg_inter_weight'])
        os.environ[f'OPTUNA_{modal_prefix}_NEG_INTER_SHIFT']  = str(modal_params[modal]['neg_inter_shift'])
        os.environ[f'OPTUNA_{modal_prefix}_POS_INTER_SHIFT']  = str(modal_params[modal]['pos_inter_shift'])
        os.environ[f'OPTUNA_{modal_prefix}_POS_INTRA_SHIFT']  = str(modal_params[modal]['pos_intra_shift'])
        os.environ[f'OPTUNA_{modal_prefix}_WEIGHT']           = str(modal_params[modal]['weight'])
    
    os.environ['OPTUNA_DATALOADER_SEED']         = str(dataloader_seed)
    os.environ['OPTUNA_DIM']                     = str(dim)
    os.environ['OPTUNA_LR']                      = str(lr)
    os.environ['OPTUNA_MAX_STEPS']               = str(args.max_steps)
    os.environ['OPTUNA_EXPERIMENT_NAME']         = f"{search_name}/trial_{trial.number}"
    os.environ['OPTUNA_EVAL_EVERY_N_EPOCHS']     = str(args.eval_every_n_epochs)
    os.environ['WANDB_MODE']                     = 'disabled'
    
    print(f"  Settings: max_steps={args.max_steps}, eval_every_n_epochs={args.eval_every_n_epochs}")


    # Always run the frozen snapshot created at the start of the search.
    snapshot_path = Path(snapshot_dir) / search_name
    train_script = snapshot_path / "src" / "train_segmentation.py"
    cmd = ['python', str(train_script), '--config-name=' + args.config_name]
    
    print(f"Training trial {trial.number}...")
    import time
    start_time = time.time()
    last_step = 0
    mious = []
    reported_steps = set()
    
    pbar = tqdm(total=args.max_steps, 
                desc=f"Trial {trial.number}", 
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n}/{total} [{elapsed}<{remaining}] {postfix}',
                ncols=100)
    
    try:
        process = subprocess.Popen(cmd, cwd=args.work_dir, stdout=subprocess.PIPE, 
                                  stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                output_str = output.strip()
                
                # Capture training progress from the Lightning log stream.
                if 'Training:' in output_str or 'global_step=' in output_str:
                    step_match = re.search(r'global_step=(\d+)', output_str)
                    if step_match:
                        current_step = int(step_match.group(1))
                        if current_step > last_step:
                            pbar.update(current_step - last_step)
                            last_step = current_step
                
                # Capture validation metrics emitted by train_segmentation.py.
                metric_match = re.search(r'OPTUNA_METRIC: step=(\d+), mIoU=([\d.]+), Accuracy=([\d.]+)', output_str)
                if metric_match:
                    step = int(metric_match.group(1))
                    miou = float(metric_match.group(2))
                    acc = float(metric_match.group(3))
                    score = miou * 0.75 + acc * 0.25
                    mious.append(score)
                    current_max = max(mious)
                    pbar.set_postfix_str(f"Score: latest={score:.4f} (mIoU={miou:.4f}, Acc={acc:.4f}), max={current_max:.4f}")
                    tqdm.write(f"  Validation at step {step}: Score={score:.4f} (mIoU={miou:.4f}, Acc={acc:.4f}), Max={current_max:.4f}")
                    
                    if step not in reported_steps:
                        trial.report(score, step)
                        reported_steps.add(step)
                        
                        if trial.should_prune():
                            tqdm.write(f"  Trial {trial.number} pruned at step {step} (Score={score:.4f})")
                            process.terminate()
                            pbar.close()
                            raise optuna.TrialPruned()
        
        pbar.close()
        if process.returncode != 0:
            print(f"Training failed with return code: {process.returncode}")
            return 0.0
    except subprocess.TimeoutExpired:
        pbar.close()
        print(f"\nTraining timeout")
        return 0.0
    except Exception as e:
        pbar.close()
        print(f"\nError: {e}")
        return 0.0
    
    if not mious:
        print(f"\n{'='*60}")
        print(f"Trial {trial.number} COMPLETED")
        print(f"  Result: No valid evaluations found")
        print(f"  Max Score: 0.0000")
        print(f"{'='*60}\n")
        max_score = 0.0
    else:
        max_score = max(mious)
        print(f"\n{'='*60}")
        print(f"Trial {trial.number} COMPLETED")
        print(f"  Max Score (0.75*mIoU + 0.25*Acc): {max_score:.4f}")
        print(f"  Evaluations: {len(mious)}")
        print(f"  Individual scores: {[f'{m:.4f}' for m in mious]}")
        print(f"{'='*60}\n")
    
    # Show the current best trial once at least one trial has completed.
    try:
        best_trial = study.best_trial
        print(f"\n{'='*60}")
        print(f"CURRENT BEST TRIAL: #{best_trial.number}")
        print(f"  Best Score (0.75*mIoU + 0.25*Acc): {best_trial.value:.4f}")
        print(f"  Parameters:")
        for key, value in best_trial.params.items():
            if key == 'lr':
                print(f"    {key}: {value:.2e}")
            elif key == 'dataloader_seed':
                print(f"    {key}: {int(value)}")
            else:
                print(f"    {key}: {value:.4f}")
        print(f"{'='*60}\n")
    except ValueError:
        pass
    
    if not args.keep_checkpoints and max_score > 0:
        cleanup_trial_checkpoints(args, search_name, trial.number)
    
    return max_score

def cleanup_trial_checkpoints(args, search_name, trial_number):
    """Remove checkpoints produced by one Optuna trial."""
    import shutil
    
    checkpoint_dir = Path(args.work_dir) / "checkpoints" / search_name / f"trial_{trial_number}"
    
    if not checkpoint_dir.exists():
        return
    
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    
    if len(checkpoint_files) == 0:
        return
    
    deleted_count = 0
    deleted_size = 0
    for ckpt_file in checkpoint_files:
        file_size = ckpt_file.stat().st_size / (1024 * 1024)
        ckpt_file.unlink()
        deleted_count += 1
        deleted_size += file_size
    
    try:
        checkpoint_dir.rmdir()
    except:
        pass
    
    if deleted_count > 0:
        print(f"  Cleaned up {deleted_count} checkpoints, freed {deleted_size:.1f} MB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=200)
    parser.add_argument('--max_steps', type=int, default=7500)
    parser.add_argument('--eval_every_n_epochs', type=int, default=1, help='Validate every N epochs')
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--work_dir', type=str, default='.')
    parser.add_argument('--config_dir', type=str, default='src/configs')
    parser.add_argument('--config_name', type=str, default='train_config_nyu.yml', help='Config file name, e.g. train_config_nyu.yml')
    parser.add_argument('--keep_checkpoints', action='store_true', help='Keep checkpoints after each trial (by default, all checkpoints are deleted to save disk space)')
    parser.add_argument('--n_startup_trials', type=int, default=5, help='Number of trials before pruning starts')
    parser.add_argument('--n_warmup_steps', type=int, default=0, help='Number of validation steps before pruning can happen')
    parser.add_argument('--search_dataloader_seed', action='store_true', help='[Override config] Search for optimal dataloader seed (0-20). If not set, uses config file setting')
    
    args = parser.parse_args()
    
    base_config_path = Path(args.config_dir) / args.config_name
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Read search_dataloader_seed from config, allow command line to override
    if args.search_dataloader_seed:
        # Command line explicitly enabled it
        search_dataloader_seed = True
    else:
        # Use config file setting (default False if not in config)
        search_dataloader_seed = base_config.get('search_dataloader_seed', False)
    
    # Store in args for easy access
    args.search_dataloader_seed = search_dataloader_seed
    
    # Read search_only_seed from config
    search_only_seed = base_config.get('search_only_seed', False)
    args.search_only_seed = search_only_seed
    
    # Validation: search_only_seed requires search_dataloader_seed to be True
    if search_only_seed and not search_dataloader_seed:
        print("\nWARNING: search_only_seed=True but search_dataloader_seed=False")
        print("  Enabling search_dataloader_seed automatically.\n")
        args.search_dataloader_seed = True
        search_dataloader_seed = True
    
    dino_version = base_config.get('dino_version', 'v1')
    model_type = base_config.get('model_type', 'vit_small')
    dino_patch_size = base_config.get('dino_patch_size', 8)
    
    modals = base_config.get('modals', ['rgb'])
    if isinstance(modals, list) and len(modals) > 0:
        modality = '_'.join(modals)
    else:
        modality = 'rgb'
    
    snapshot_dir = Path(args.work_dir) / "optuna" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = 'small' if 'small' in model_type else 'base'
    
    search_name = f"UniM2_search_{modality}_dino{dino_version}_{model_name}_patch{dino_patch_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    snapshot_path = create_code_snapshot(args.work_dir, snapshot_dir, search_name)
    
    db_dir = Path(args.work_dir) / "optuna" / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / f"{search_name}.db"
    storage = f"sqlite:///{db_path}"
    
    print(f"\n{'='*60}")
    print(f"Starting Hyperparameter Search")
    print(f"  Search Name: {search_name}")
    print(f"  Modality: {modality}")
    print(f"  DINO Version: {dino_version}")
    print(f"  Model Type: {model_type} (patch_size={dino_patch_size})")
    print(f"  Trials: {args.n_trials}")
    print(f"  Max Steps: {args.max_steps}")
    print(f"  Eval Every N Epochs: {args.eval_every_n_epochs}")
    print(f"  Pruning: Enabled (startup_trials={args.n_startup_trials}, warmup_steps={args.n_warmup_steps})")
    print(f"  Search DataLoader Seed: {'Yes (0-20)' if args.search_dataloader_seed else 'No (fixed at 7)'}")
    if args.search_only_seed:
        print(f"  Search Mode: ONLY SEED (all other params fixed)")
    else:
        print(f"  Search Mode: All hyperparameters")
    print(f"  Database: {db_path}")
    print(f"{'='*60}\n")
    
    # Prune trials whose intermediate scores fall below the configured percentile.
    study = optuna.create_study(
        study_name=search_name,
        storage=storage,
        load_if_exists=False,
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=PercentilePruner(
            percentile=50.0,
            n_startup_trials=args.n_startup_trials,
            n_warmup_steps=args.n_warmup_steps,
            interval_steps=1
        )
    )
    
    study.optimize(lambda trial: objective(trial, args, base_config, search_name, study, snapshot_dir), n_trials=args.n_trials)
    
    print(f"\nCleaning up code snapshot...")
    snapshot_path = Path(snapshot_dir) / search_name
    if snapshot_path.exists():
        shutil.rmtree(snapshot_path)
        print(f"  Snapshot removed: {snapshot_path}")
    
    print(f"\n{'='*60}")
    print("Best trial:")
    print(f"  Value: {study.best_trial.value:.4f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        if key == 'lr':
            print(f"    {key}: {value:.2e}")
        elif key == 'dataloader_seed':
            print(f"    {key}: {int(value)}")
        else:
            print(f"    {key}: {value:.4f}")
    print(f"{'='*60}\n")
    
    result_filename = f"UniM2_best_params_{modality}_dino{dino_version}_{model_name}_patch{dino_patch_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    result_path_config = Path(args.config_dir) / result_filename
    
    results_dir = Path(args.work_dir) / "optuna_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path_main = results_dir / result_filename
    
    report_content = []
    report_content.append(f"{'='*80}\n")
    report_content.append(f"UniM2 Optuna Hyperparameter Search Results\n")
    report_content.append(f"{'='*80}\n\n")
    
    report_content.append(f"Search Information:\n")
    report_content.append(f"  Search Name: {search_name}\n")
    report_content.append(f"  Modality: {modality}\n")
    report_content.append(f"  DINO Version: {dino_version}\n")
    report_content.append(f"  Model Type: {model_type} (patch_size={dino_patch_size})\n")
    report_content.append(f"  Database: {db_path}\n")
    report_content.append(f"  Total Trials: {len(study.trials)}\n")
    report_content.append(f"  Completed Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}\n")
    report_content.append(f"  Pruned Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n\n")
    
    report_content.append(f"Best Trial (#{study.best_trial.number}):\n")
    report_content.append(f"  Score (0.75*mIoU + 0.25*Acc): {study.best_trial.value:.4f}\n\n")
    
    report_content.append(f"Best Parameters:\n")
    report_content.append(f"-" * 80 + "\n")
    for key, value in study.best_trial.params.items():
        if key == 'lr':
            report_content.append(f"  {key}: {value:.2e}\n")
        elif key == 'dataloader_seed':
            report_content.append(f"  {key}: {int(value)}\n")
        else:
            report_content.append(f"  {key}: {value:.4f}\n")
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 1:
        report_content.append(f"\n\nAll Completed Trials (sorted by score):\n")
        report_content.append(f"-" * 80 + "\n")
        sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)
        for i, trial in enumerate(sorted_trials[:10]):
            report_content.append(f"\n  Rank {i+1} - Trial #{trial.number}:\n")
            report_content.append(f"    Score: {trial.value:.4f}\n")
            report_content.append(f"    Parameters:\n")
            for key, value in trial.params.items():
                if key == 'lr':
                    report_content.append(f"      {key}: {value:.2e}\n")
                elif key == 'dataloader_seed':
                    report_content.append(f"      {key}: {int(value)}\n")
                else:
                    report_content.append(f"      {key}: {value:.4f}\n")
    
    report_content.append(f"\n{'='*80}\n")
    report_content.append(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report_content.append(f"{'='*80}\n")
    
    report_text = ''.join(report_content)
    
    with open(result_path_config, 'w') as f:
        f.write(report_text)
    with open(result_path_main, 'w') as f:
        f.write(report_text)
    
    print(f"\nResults saved to:")
    print(f"  - {result_path_config}")
    print(f"  - {result_path_main}")

if __name__ == "__main__":
    main()
