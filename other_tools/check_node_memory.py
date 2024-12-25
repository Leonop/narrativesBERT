#!/usr/bin/env python3
import subprocess
import pandas as pd

def get_node_memory():
    """Get memory information for all nodes in the cluster"""
    try:
        # Try different commands to get memory information
        commands = [
            "sinfo -N -o '%N|%e|%m|%C|%O'",  # Standard format
            "sinfo -N -o '%N|%e|%M|%C|%O'",  # Using %M for memory in MB
            "scontrol show node"  # Detailed node information
        ]
        
        print("Attempting to get memory information...")
        
        # Try scontrol first for detailed information
        result = subprocess.run("scontrol show node", shell=True, capture_output=True, text=True)
        print("\nRaw scontrol output (first 500 chars):")
        print(result.stdout[:500])
        
        # Also try free command if possible
        try:
            free_result = subprocess.run("free -g", shell=True, capture_output=True, text=True)
            print("\nSystem memory (free -g):")
            print(free_result.stdout)
        except:
            pass
        
        # Try each sinfo command
        for cmd in commands:
            print(f"\nTrying command: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print("Output:")
            print(result.stdout)
            
            if result.returncode == 0 and result.stdout.strip():
                break
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd)

        # Parse scontrol output for memory information
        nodes_info = []
        current_node = {}
        
        for line in result.stdout.split('\n'):
            if line.startswith('NodeName='):
                if current_node:
                    nodes_info.append(current_node)
                current_node = {}
                
                # Parse key information
                for item in line.split():
                    if '=' in item:
                        key, value = item.split('=', 1)
                        if key in ['NodeName', 'RealMemory', 'State']:
                            current_node[key] = value
        
        if current_node:
            nodes_info.append(current_node)
        
        if not nodes_info:
            print("Warning: No node information could be parsed!")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(nodes_info)
        
        print("\nParsed node information:")
        print(df)
        
        # Convert memory to GB (RealMemory is typically in MB)
        if 'RealMemory' in df.columns:
            df['memory_gb'] = pd.to_numeric(df['RealMemory'], errors='coerce') / 1024
        else:
            print("Warning: No RealMemory information found!")
            return None
        
        # Calculate statistics
        stats = {
            'min_memory': df['memory_gb'].min(),
            'max_memory': df['memory_gb'].max(),
            'mean_memory': df['memory_gb'].mean(),
            'median_memory': df['memory_gb'].median()
        }
        
        print("\nMemory statistics (GB):")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")
        
        return {
            'stats': stats,
            'available_nodes': df[df['State'].str.contains('IDLE', case=False, na=False)],
            'all_nodes': df
        }
    
    except Exception as e:
        print(f"Error getting node memory information: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
        return None

def suggest_memory_request(memory_info, dataset_size_gb=None):
    """Suggest memory request based on available resources and dataset size"""
    if not memory_info:
        print("No memory information available, using default value")
        return 128  # Increased default value
    
    stats = memory_info['stats']
    print(f"\nMemory statistics found: {stats}")  # Debug print
    
    # If dataset size is provided, use it as a base
    if dataset_size_gb:
        # Request 2x dataset size or median available memory, whichever is smaller
        suggested_memory = min(dataset_size_gb * 2, stats['median_memory'])
    else:
        # Otherwise use 75% of median available memory
        suggested_memory = stats['median_memory'] * 0.75
    
    # Round to nearest multiple of 8
    suggested_memory = round(suggested_memory / 8) * 8
    
    # Cap at maximum available
    suggested_memory = min(suggested_memory, stats['max_memory'] * 0.9)
    
    # Ensure minimum reasonable value
    suggested_memory = max(suggested_memory, 64)
    
    return int(suggested_memory)

if __name__ == "__main__":
    print("Starting memory check...")
    # Get memory information
    memory_info = get_node_memory()
    
    if memory_info:
        print("\nCluster Memory Statistics (GB):")
        for key, value in memory_info['stats'].items():
            print(f"{key}: {value:.2f}")
        
        print("\nAvailable Nodes:")
        print(memory_info['available_nodes'])
        
        # Example with dataset size of 50GB
        suggested_mem = suggest_memory_request(memory_info, dataset_size_gb=50)
        print(f"\nSuggested memory request: {suggested_mem}GB")
    else:
        print("Could not get memory information, using default value of 128GB") 