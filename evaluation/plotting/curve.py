import matplotlib.pyplot as plt

def plot_metric_curve1(plot_path, train_values, val_values, val_iterations, metric_name,
                    xlabel="Iteration", eval_interval=100):
    
    """Trace and save a train/val curve for any metrics.

    Args:
        train_values: liste des valeurs train à chaque itération
        val_values: liste des valeurs val aux eval points
        val_iterations: liste des itérations où val a été calculée
        metric_name: str, nom de la métrique pour titre/labels, ex: "Loss"
        save_name: str, nom du fichier sans extension, ex: "loss", "accuracy"
        xlabel: str, label axe X
        eval_interval: int, fallback si val_iterations vide"""
    
    if len(train_values) == 0:
        raise ValueError(f'No training data to plot for {metric_name}')

    plt.figure(figsize=(10, 6))

    # Training curve à chaque itération
    plt.plot(range(1, len(train_values) + 1), train_values,
             label=f'Train {metric_name}', linewidth=1.5)

    # Validation curve aux points d'eval
    if len(val_values) > 0:
        if len(val_iterations) > 0 and len(val_iterations) == len(val_values):
            # On utilise les vraies itérations d'eval
            plt.plot(val_iterations, val_values, 'o-',
                    label=f'Val {metric_name}', linewidth=1.5, markersize=6, alpha=0.7)
        else:
            # Fallback: on suppose eval_interval constant
            val_iter_positions = [eval_interval * (i + 1) for i in range(len(val_values))]
            plt.plot(val_iter_positions, val_values, 'o-',
                    label=f'Val {metric_name}', linewidth=1.5, markersize=6, alpha=0.7)

    plt.title(f'Model {metric_name} During Training', fontsize=14)
    plt.ylabel(metric_name, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


def plot_metric_curve2(plot_path, train_values, val_values, val_iterations, metric_name,
                      xlabel="Epoch", eval_interval=100):
    if len(train_values) == 0:
        raise ValueError(f'No training data to plot for {metric_name}')

    plt.figure(figsize=(10, 6))

    # La courbe train est toujours aux points d'eval (1 valeur par eval_interval)
    # On génère les itérations correspondantes
    train_iterations = [eval_interval * (i + 1) for i in range(len(train_values))]
    plt.plot(train_iterations, train_values,
             label=f'Train {metric_name}', linewidth=1.5)

    if len(val_values) > 0:
        if len(val_iterations) == len(val_values):
            x_val = val_iterations
        else:
            x_val = [eval_interval * (i + 1) for i in range(len(val_values))]
        plt.plot(x_val, val_values, 'o-',
                 label=f'Val {metric_name}', linewidth=1.5, markersize=6, alpha=0.7)

    plt.title(f'Model {metric_name} During Training', fontsize=14)
    plt.ylabel(metric_name, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path
