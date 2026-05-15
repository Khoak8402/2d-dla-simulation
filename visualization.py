import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import dla_simulation as dla

def display_aggregate_grow(results):
    """Displays snapshots of the aggregate at 10%, 50%, and 100% completion."""
    snapshots = results['snapshots']
    glued_counts = results['glued_counts']
    total_particles = results['n_particles']
    
    # Calculate target particle counts
    targets = [0.10 * total_particles, 0.50 * total_particles, total_particles]
    
    # Find the snapshot index that is closest to each target count
    indices = [np.argmin(np.abs(glued_counts - target)) for target in targets]
    titles = ["10% Progress", "50% Progress", "100% Progress"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"DLA Aggregate Growth ({total_particles} total particles)", 
                 fontsize=16, fontweight='bold', y=0.95)
    
    for ax, idx, title in zip(axes, indices, titles):
        binary_map = np.where(snapshots[idx] == 2, 1, 0)
        
        ax.imshow(binary_map, cmap='magma', interpolation='nearest')
        ax.set_title(f"{title}\n({glued_counts[idx]} particles)", fontsize=12)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()
    
    
def display_mass_radius_plot(results):
    """Displays the Log(R) vs Log(M) plot with discrete points, fit line, and theory line."""
    radii = results['radii']
    masses = results['masses']
    
    max_radius = np.max(radii)
    valid_id = (masses > 10) & (radii < max_radius * 0.8)
    
    if np.sum(valid_id) > 5:
        log_r = np.log(radii[valid_id])
        log_m = np.log(masses[valid_id])
        
        Df, C = np.polyfit(log_r, log_m, 1)
        
        fit_line = Df * log_r + C
        theory_line = 1.71 * log_r + C 
        
        # Create the plot
        plt.figure(figsize=(8, 6))
        
        plt.plot(log_r, log_m, 'o', color='royalblue', label='Measured Data', alpha=0.7)
        
        plt.plot(log_r, fit_line, 'r-', linewidth=2.5, label=f'Simulation Fit ($D_f$ = {Df:.3f})')
        
        plt.plot(log_r, theory_line, 'g--', linewidth=2.5, label='Theory ($D_f$ = 1.71)')
        
        plt.title("Mass-Radius Relationship (Log-Log Scale)", fontsize=14, fontweight='bold')
        plt.xlabel("$log(R)$", fontsize=12)
        plt.ylabel("$log(M)$", fontsize=12)
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.show()
    else:
        print("Error: Not enough valid datapoints for fitting!")


def display_grow_progress_plot(results):
    """Displays a line plot showing how the aggregate grew over time."""
    time_history = results['time-history']
    growth_history = results['growth-history']
    total_particles = results['n_particles']
    
    plt.figure(figsize=(8, 5))
    
    # Plot the growth curve
    plt.plot(time_history, growth_history, color='royalblue', linewidth=2.5)
    
    plt.axhline(y=total_particles, color='gray', linestyle=':', alpha=0.8, 
                label=f'Target ({total_particles} particles)')
    
    plt.title("Aggregate Growth Progress", fontsize=14, fontweight='bold')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Number of Glued Particles", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10, loc='lower right')
    
    plt.tight_layout()
    plt.show()


# Test
results = dla.simulation(N=512, n_seeds=1, n_walkers=10000, max_steps=50000, 
                      reinject_timeout=1000, inj_mode='radial', inj_radius= 160)
display_aggregate_grow(results)
display_mass_radius_plot(results)
display_grow_progress_plot(results)