import os

import numpy as np
import numba
from numba import jit

@jit(nopython= True, cache=True)
def setup_compute_df(grid, N, center_x, center_y, max_radius):
    """Setup the radii and mass parameters for computing the fractal dimension."""
    # Number of circle drawed to take sampling, maximum 50
    n_radii= min(50, max_radius)
    # Radius of these circles
    radii= np.logspace(0.5, np.log10(max_radius), n_radii)
    # Sampling mass
    masses= np.zeros(n_radii)
    
    for ir, r in enumerate(radii):
        mass= 0
        for i in range(N):
            for j in range(N):
                if grid[i, j] == 2: # Counting immoblie walkers inside 
                    sq_dist= (i-center_x)**2 + (j-center_y)**2
                    if sq_dist < r**2:
                        mass+= 1
        masses[ir]= mass
        
    return radii, masses

@jit(nopython= True, cache=True)
def aggregate_bounds(grid, N):
    """Return the aggregate bound in the grid"""
    # Find the boundary rectangler of the aggregate
    min_x, max_x= N, 0
    min_y, max_y= N, 0
    
    for i in range(N):
        for j in range(N):
            if grid[i, j] == 2:
                min_x= min(min_x, i)
                max_x= max(max_x, i)
                min_y= min(min_y, j)
                max_y= max(max_y, j)
                
    return min_x, max_x, min_y, max_y


@jit(nopython= True, cache=True)
def random_injected(i, x, y, grid, N, min_x, max_x, min_y, max_y):
        """Randomly injected the walker outside the aggregate"""
        margin = 20 # The margin for seperate the spawn space around the aggregate
        # Walker must be spawn far enough from the aggregate center  
        x_range = max(max_x - min_x + 2 * margin, N // 4)
        y_range = max(max_y - min_y + 2 * margin, N // 4)
        
        # Center of the aggregate
        cx = (min_x + max_x) // 2
        cy = (min_y + max_y) // 2
        
        max_iter= 25
        for _ in range(max_iter):
            # Spawn the walker randomly on computed spawn space
            x[i] = (cx + np.random.randint(-x_range//2, x_range//2)) % N
            y[i] = (cy + np.random.randint(-y_range//2, y_range//2)) % N
            if grid[x[i], y[i]] == 0:
               break
            
        # If failed to find an empty site, accept overlapping this turn 
        grid[x[i], y[i]] = 1


@jit(nopython= True, cache=True)
def radial_injected(i, x, y, grid, N, center_x, center_y, inj_radius):
        """Radially injected the walker in the circle with radius R_inj"""
        # For the radial injection, walkers will be respawn randomly in the circle with radius R_inj
        theta = 2.0 * np.pi * np.random.random()
        x[i] = int(center_x + inj_radius * np.cos(theta)) % N
        y[i] = int(center_y + inj_radius * np.sin(theta)) % N
            
         # Make sure position is empty
        max_attempts = 20
        for _ in range(max_attempts):
           if grid[x[i], y[i]] == 0:
            break
           x[i] = np.random.randint(0, N)
           y[i] = np.random.randint(0, N)
    
        grid[x[i], y[i]] = 1


@jit(nopython= True, cache=True)
def reinject_walker(i, x, y, grid, N, min_x, max_x, min_y, max_y, 
                    inj_mode, inj_radius, center_x, center_y):
    """Reinject walkers that are wandered for so long"""
    #Remove it from current site
    grid[x[i], y[i]]= 0
    
    if inj_mode == "random":
        random_injected(i, x, y, grid, N, min_x, max_x, min_y, max_y)
    
    elif inj_mode == "radial":
        radial_injected(i, x, y, grid, N, center_x, center_y, inj_radius)
    
    
@jit(nopython=True, cache=True)
def random_walk_step(x, y, status, grid,
                    walker_age, N , n_walkers ,reinject_timeout, min_x, max_x, min_y, max_y,
                    inj_mode, inj_radius, center_x, center_y):
    """
    Perform one random walk step for all mobile particles.
    Returns number of newly stuck particles.
    """
    n_glued = 0
    
    # Direction arrays 
    dx = np.array([-1, 0, 1, 0])
    dy = np.array([0, -1, 0, 1])
    
    # Create temporary grid to avoid conflicts
    new_grid = grid.copy()
    
    for i in range(n_walkers):
        if status[i] == 1:  # Mobile particle
            walker_age[i] += 1
            
            # Re-inject if walker is taking too long
            if reinject_timeout > 0 and walker_age[i] > reinject_timeout:
                reinject_walker(i, x, y, new_grid, N, 
                               min_x, max_x, min_y, max_y,
                               inj_mode, inj_radius,
                               center_x, center_y)
                walker_age[i] = 0
                continue
            
            # Clear old position
            new_grid[x[i], y[i]] = 0
            
            # Pick random direction
            direction = np.random.randint(0, 4)
            
            # Calculate new position 
            x_new = (x[i] + dx[direction]) % N
            y_new = (y[i] + dy[direction]) % N
            
            # Check if new position is already occupied by another mobile walker
            if new_grid[x_new, y_new] == 1:
                new_grid[x[i], y[i]] = 1
                continue
            
            # Check for sticky neighbors
            has_sticky_neighbor = False
            for d in range(4):
                nx = (x_new + dx[d]) % N
                ny = (y_new + dy[d]) % N
                if grid[nx, ny] == 2:  
                    has_sticky_neighbor = True
                    break
            
            if has_sticky_neighbor:
                # Stick particle
                new_grid[x_new, y_new] = 2
                status[i] = 2
                n_glued += 1
                x[i] = x_new
                y[i] = y_new
                walker_age[i] = 0
            else:
                # Move particle
                new_grid[x_new, y_new] = 1
                x[i] = x_new
                y[i] = y_new
    
    # Copy new grid back
    for i in range(N):
        for j in range(N):
            grid[i, j] = new_grid[i, j]
    
    return n_glued   


@jit(nopython=True, cache=True)
def initialize_radial_injection(N, n_walkers, inj_radius, grid, center_x, center_y):
    """Initialize walkers on a circle for radial injection."""
    x = np.zeros(n_walkers, dtype=np.int32)
    y = np.zeros(n_walkers, dtype=np.int32)
    
    for i in range(n_walkers):
        theta = 2.0 * np.pi * np.random.random()
        x[i] = int(center_x + inj_radius * np.cos(theta)) % N
        y[i] = int(center_y + inj_radius * np.sin(theta)) % N
        grid[x[i], y[i]] = 1  # Mark as occupied
    
    return x, y


@jit(nopython=True, cache=True)
def initialize_random_walkers(N, n_walkers, grid):
    """Initialize walkers randomly ensuring no overlap."""
    x = np.zeros(n_walkers, dtype=np.int32)
    y = np.zeros(n_walkers, dtype=np.int32)
    
    for i in range(n_walkers):
        # Find empty position
        max_attempts = 100
        for _ in range(max_attempts):
            pos_x = np.random.randint(0, N)
            pos_y = np.random.randint(0, N)
            
            if grid[pos_x, pos_y] == 0:
                x[i] = pos_x
                y[i] = pos_y
                grid[pos_x, pos_y] = 1
                break
        else:
            # If no empty slot found, accept overlap 
            x[i] = np.random.randint(0, N)
            y[i] = np.random.randint(0, N)
            grid[x[i], y[i]] = 1
    
    return x, y

def flood_fill(grid, visited, i, j):
    """Flood fill to mark connected components."""
    stack = [(i, j)]
        
    while stack:
        x, y = stack.pop()
            
        if x < 0 or x >= grid.shape[0] or y < 0 or y >= grid.shape[1]:
            continue
        if visited[x, y] or grid[x, y] != 2:
            continue
            
        visited[x, y] = True
            
        stack.extend([
            (x-1, y), (x+1, y), (x, y-1), (x, y+1)
        ])

def count_aggregate(grid, N):
        """Count number of separate aggregates using flood fill."""
        visited = np.zeros_like(grid, dtype=bool)
        n_aggregates = 0
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 2 and not visited[i, j]:
                    flood_fill(grid, visited, i, j)
                    n_aggregates += 1
        
        return n_aggregates
    

def initialize(N, n_seeds, n_walkers, max_steps, reinject_timeout, inj_mode, inj_radius):
    """"Initialize the grid, seeds, and walkers prior to the simulation."""
    # CPU core config
    default_cpu_cores= os.cpu_count()
    numba.set_num_threads(default_cpu_cores)
    #Snapshot interval (default 1000)
    snapshot_interval= 1000
    # Reinjection timeout
    if reinject_timeout is None:
        reinject_timeout = max(N*2, 1000)
    # Max iterations
    max_iter= min(max_steps, n_walkers*N//2)
    
    #Initialize grid
    grid = np.zeros((N, N), dtype=np.int32)
    
    #Initialize particles
    x = np.zeros(n_walkers, dtype=np.int32)
    y = np.zeros(n_walkers, dtype=np.int32)
    status = np.ones(n_walkers, dtype=np.int32)  # 1 for mobile, 2 for stuck
    walker_age = np.zeros(n_walkers, dtype=np.int32)  # Age of each walker
    
    #Initialize seeds' positions in the lattice
    if n_seeds == 1: #Single seed 
        center_x, center_y = N // 2, N // 2
        grid[center_x, center_y] = 2
    else: #Multiple seeds
        for _ in range(n_seeds):
            seed_x = np.random.randint(0, N)
            seed_y = np.random.randint(0, N)
            while grid[seed_x, seed_y] != 0:  # Ensure no overlap 
                seed_x = np.random.randint(0, N)
                seed_y = np.random.randint(0, N)
            grid[seed_x, seed_y] = 2
        center_x, center_y = N // 2, N // 2  
        
    # Initialize walkers based on injection mode
    if inj_mode is None or inj_mode == "random":
        x, y = initialize_random_walkers(N, n_walkers, grid)
    elif inj_mode == "radial":
        if inj_radius is None:
            inj_radius = N // 4  # Default R_inj
        x, y = initialize_radial_injection(N, n_walkers, inj_radius, grid, center_x, center_y)
        
    return N, grid, x, y, status, walker_age, center_x, center_y, max_iter, reinject_timeout, inj_radius, snapshot_interval


def simulation(N, n_seeds, n_walkers, max_steps, reinject_timeout, inj_mode, inj_radius):
    """Run the DLA simulation with specified parameters.
    Returns the final grid and the number of glued particles at each step."""
    
    #Initialize
    N, grid, x, y, status, walker_age, center_x, center_y, max_iter, reinject_timeout, inj_radius, snapshot_interval = initialize(N, n_seeds, n_walkers, max_steps, reinject_timeout, inj_mode, inj_radius)
    
    # Tracking values
    n_glued_total= n_seeds
    iters= 0
    time_history= []
    growth_history= []
    snapshots= [grid.copy()]
    glued_counts= [n_seeds]
    
    # Simulation loop
    last_snapshot= n_seeds
    update_every_n_iters= 100
    min_x, max_x, min_y, max_y= aggregate_bounds(grid, N)
    
    while n_glued_total < n_walkers and iters < max_iter:
        if iters % update_every_n_iters == 0:
            min_x, max_x, min_y, max_y= aggregate_bounds(grid, N)
        
        if iters % 500 == 0:
            time_history.append(iters)
            growth_history.append(n_glued_total)
        
        n_glued_this_step= random_walk_step(x, y, status, grid,
                                            walker_age, N , n_walkers ,reinject_timeout, min_x, max_x, min_y, max_y, 
                                            inj_mode, inj_radius if inj_mode =='radial' else 0.0, center_x, center_y)
        n_glued_total += n_glued_this_step
        iters += 1
        if n_glued_total - last_snapshot >= snapshot_interval:
            snapshots.append(grid.copy())
            glued_counts.append(n_glued_total)
            last_snapshot= n_glued_total
        
    # Final snapshot at the end of simulation
    if glued_counts[-1] != n_glued_total:
        snapshots.append(grid.copy())
        glued_counts.append(n_glued_total)
        
    # Compute fractal dimension 
    final_grid= grid
    max_radius= min(N//2, 200)
    
    radii, masses= setup_compute_df(final_grid, N, center_x, center_y, max_radius)
    
    # Utilize linear regression to compute the slope of log-log plot
    # Removing small data points to avoid noise
    valid_id= (masses > 10) & (radii < max_radius*0.8)
    if np.sum(valid_id) > 10: # Ensure enough data
        log_r= np.log(radii[valid_id])
        log_m= np.log(masses[valid_id])
        Df= np.polyfit(log_r, log_m, 1)[0]
    else:
        Df= np.nan  # Not enough data 
    
    #Count final grid aggregate
    n_aggregate= count_aggregate(final_grid, N)
    
    return { # Simulation results
        'grid' : final_grid,
        'snapshots' : np.array(snapshots),
        'glued_counts' : np.array(glued_counts),
        'time-history' : np.array(time_history),
        'growth-history' : np.array(growth_history),
        'fractal_dimension' : Df,
        'n_aggregate' : n_aggregate,
        'n_particles' : n_glued_total,
        'center': (center_x, center_y),
        'radii': radii,
        'masses': masses
    }
    
        
        

    
    
        
    
            