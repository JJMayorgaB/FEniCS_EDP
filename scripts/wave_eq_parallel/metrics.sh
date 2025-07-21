#!/bin/bash

# Wave Equation Parallel Performance Metrics Script
# 
# This script automatically tests the wave equation solver with different
# core counts and calculates parallel performance metrics.

echo "======================================================="
echo "    Wave Equation Parallel Performance Metrics      "
echo "======================================================="
echo

# Check if required files exist
SCRIPT_NAME="wave_eq_parallel.py"
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Error: $SCRIPT_NAME not found in current directory"
    echo "Please make sure you have the wave equation script"
    exit 1
fi

# Check if mpirun is available
if ! command -v mpirun &> /dev/null; then
    echo "Error: mpirun not found. Please install OpenMPI or MPICH"
    echo "On Ubuntu/Debian: sudo apt install openmpi-bin libopenmpi-dev"
    echo "On CentOS/RHEL: sudo yum install openmpi openmpi-devel"
    exit 1
fi

# Check if bc (calculator) is available for floating point operations
if ! command -v bc &> /dev/null; then
    echo "Error: bc (calculator) not found. Please install bc"
    echo "On Ubuntu/Debian: sudo apt install bc"
    echo "On CentOS/RHEL: sudo yum install bc"
    exit 1
fi

# Create results directory
RESULTS_DIR="metrics_results"
mkdir -p "$RESULTS_DIR"

# Results file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/wave_performance_$TIMESTAMP.txt"

echo "Results will be saved to:"
echo "  - Text format: $RESULTS_FILE"
echo

# Initialize result files
cat > "$RESULTS_FILE" << EOF
EOF

# Store results for analysis
declare -A times
declare -A speedups
declare -A efficiencies

# Function to run single benchmark
run_benchmark() {
    local cores=$1
    echo -n "Testing with $cores cores... "
    
    # Run the benchmark with timeout and capture both stdout and stderr
    local temp_output=$(mktemp)
    local start_time=$(date +%s.%N)
    
    timeout 1200 mpirun -np $cores python3 "$SCRIPT_NAME" > "$temp_output" 2>&1
    local exit_code=$?
    local end_time=$(date +%s.%N)
    
    if [ $exit_code -eq 0 ]; then
        # Extract execution time from output
        local exec_time=$(grep "Execution completed in" "$temp_output" | awk '{print $4}')
        
        if [ -n "$exec_time" ] && [ $(echo "$exec_time > 0" | bc -l) -eq 1 ]; then
            times[$cores]=$exec_time
            echo "✓ ${exec_time}s"
            rm "$temp_output"
            return 0
        else
            echo "✗ Could not parse execution time"
            echo "Output preview:" 
            head -10 "$temp_output" | sed 's/^/    /'
            rm "$temp_output"
            return 1
        fi
    elif [ $exit_code -eq 124 ]; then
        echo "✗ Timeout (>10 min)"
        rm "$temp_output"
        return 1
    else
        echo "✗ Failed (exit code: $exit_code)"
        echo "Error details:"
        tail -5 "$temp_output" | sed 's/^/    /'
        rm "$temp_output"
        return 1
    fi
}

# Function to display progress bar
show_progress() {
    local current=$1
    local total=$2
    local width=30
    local percentage=$((current * 100 / total))
    local completed=$((current * width / total))
    
    printf "\rProgress: ["
    for ((i=0; i<width; i++)); do
        if [ $i -lt $completed ]; then
            printf "█"
        else
            printf "░"
        fi
    done
    printf "] %d%% (%d/%d)" $percentage $current $total
}

# Test configuration
core_counts=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
max_cores=$(nproc 2>/dev/null || echo "unknown")

echo "System information:"
echo "  - Available cores: $max_cores"
echo "  - Testing cores: ${core_counts[*]}"
echo "  - Timeout per test: 10 minutes"
echo

# Verify we don't exceed available cores
if [ "$max_cores" != "unknown" ]; then
    max_test_cores=${core_counts[-1]}
    if [ $max_test_cores -gt $max_cores ]; then
        echo "Warning: Testing with $max_test_cores cores but only $max_cores available"
        echo "This may lead to oversubscription and poor performance"
    fi
fi

# Store results for analysis
declare -A times
successful_runs=0
total_runs=${#core_counts[@]}

echo "Starting metrics runs..."
echo

# Run metrics with progress tracking
for i in "${!core_counts[@]}"; do
    cores=${core_counts[$i]}
    show_progress $i $total_runs
    echo
    
    if run_benchmark $cores; then
        ((successful_runs++))
    fi
    echo
done

show_progress $total_runs $total_runs
echo
echo
echo "Completed $successful_runs/$total_runs benchmark runs"

# Calculate and display metrics
if [ $successful_runs -ge 2 ]; then
    echo
    echo "======================================================="
    echo "                Performance Analysis                   "
    echo "======================================================="
    
    # Get baseline time (prefer 1 core, fallback to minimum time)
    base_time=""
    base_cores=""
    
    if [ -n "${times[1]}" ]; then
        base_time=${times[1]}
        base_cores=1
    else
        # Find minimum time and corresponding cores as baseline
        for cores in "${!times[@]}"; do
            if [ -z "$base_time" ] || [ $(echo "${times[$cores]} < $base_time" | bc -l) -eq 1 ]; then
                base_time=${times[$cores]}
                base_cores=$cores
            fi
        done
    fi
    
    echo "Baseline: $base_cores cores, ${base_time}s"
    echo
    
    # Table header
    printf "%-6s | %-10s | %-8s | %-12s | %-10s\n" "Cores" "Time(s)" "Speedup" "Efficiency%" "Notes"
    printf "%-6s-|-%-10s-|-%-8s-|-%-12s-|-%-10s\n" "------" "----------" "--------" "------------" "----------"
    
    # Calculate and display metrics for each run
    declare -A speedups
    declare -A efficiencies
    
    for cores in $(printf '%s\n' "${!times[@]}" | sort -n); do
        exec_time=${times[$cores]}
        
        # Calculate speedup and efficiency
        speedup=$(echo "scale=2; $base_time / $exec_time" | bc -l)
        efficiency=$(echo "scale=1; ($speedup / $cores) * 100" | bc -l)
        
        speedups[$cores]=$speedup
        efficiencies[$cores]=$efficiency
        
        # Determine performance notes
        notes=""
        if [ $cores -eq $base_cores ]; then
            notes="(baseline)"
        elif [ $(echo "$efficiency > 90" | bc -l) -eq 1 ]; then
            notes="excellent"
        elif [ $(echo "$efficiency > 75" | bc -l) -eq 1 ]; then
            notes="good"
        elif [ $(echo "$efficiency > 50" | bc -l) -eq 1 ]; then
            notes="fair"
        else
            notes="poor"
        fi
        
        printf "%-6d | %-10.4f | %-8.2f | %-12.1f | %-10s\n" $cores $exec_time $speedup $efficiency "$notes"
        
        # Save to files (format: cores speedup efficiency)
        printf "%5d %8.2f %10.1f\n" $cores $speedup $efficiency >> "$RESULTS_FILE"
    done
    
    echo
    echo "======================================================="
    echo "                     Summary                           "
    echo "======================================================="
    
    # Find best performance metrics
    max_speedup=0
    max_efficiency=0
    best_speedup_cores=""
    best_efficiency_cores=""
    
    for cores in "${!speedups[@]}"; do
        if [ $(echo "${speedups[$cores]} > $max_speedup" | bc -l) -eq 1 ]; then
            max_speedup=${speedups[$cores]}
            best_speedup_cores=$cores
        fi
        if [ $(echo "${efficiencies[$cores]} > $max_efficiency" | bc -l) -eq 1 ]; then
            max_efficiency=${efficiencies[$cores]}
            best_efficiency_cores=$cores
        fi
    done
    
    echo "Best speedup:    ${max_speedup}x at $best_speedup_cores cores"
    echo "Best efficiency: ${max_efficiency}% at $best_efficiency_cores cores"
    
    # Scalability assessment at maximum cores tested
    max_cores_tested=$(printf '%s\n' "${!times[@]}" | sort -n | tail -1)
    final_speedup=${speedups[$max_cores_tested]}
    final_efficiency=${efficiencies[$max_cores_tested]}
    
    echo "At $max_cores_tested cores: ${final_speedup}x speedup, ${final_efficiency}% efficiency"
    
    # Overall scalability assessment
    if [ $(echo "$final_efficiency > 80" | bc -l) -eq 1 ]; then
        scalability="Excellent"
        color="\033[0;32m" # Green
    elif [ $(echo "$final_efficiency > 60" | bc -l) -eq 1 ]; then
        scalability="Good"
        color="\033[0;33m" # Yellow
    elif [ $(echo "$final_efficiency > 40" | bc -l) -eq 1 ]; then
        scalability="Fair"
        color="\033[0;35m" # Magenta
    else
        scalability="Poor"
        color="\033[0;31m" # Red
    fi
    
    printf "Overall scalability: ${color}${scalability}\033[0m\n"
    
    # Recommendations
    echo
    echo "Recommendations:"
    if [ $(echo "$final_efficiency > 75" | bc -l) -eq 1 ]; then
        echo "  ✓ Code scales well - consider using more cores for larger problems"
    elif [ $(echo "$final_efficiency > 50" | bc -l) -eq 1 ]; then
        echo "  ○ Moderate scaling - optimal performance around $best_efficiency_cores-$best_speedup_cores cores"
    else
        echo "  ⚠ Poor scaling - check for bottlenecks, consider smaller core counts"
        echo "  ⚠ May benefit from problem size scaling or algorithm optimization"
    fi
    
    # Save summary to results file
    {
        echo ""
    } >> "$RESULTS_FILE"
    
else
    echo "======================================================="
    echo "              Insufficient Data                        "
    echo "======================================================="
    echo "Need at least 2 successful runs for performance analysis"
    echo "Only $successful_runs successful runs completed"
fi

echo
echo "======================================================="
echo "                    Files Generated                    "
echo "======================================================="
echo "Results saved to:"
echo "  📄 $RESULTS_FILE"
echo
echo "To analyze results further:"
echo "  - View detailed results: cat '$RESULTS_FILE'"
echo
echo "======================================================="
echo "                 Benchmark Complete!                   "
echo "======================================================="

# Display quick summary table for easy copy-paste
if [ $successful_runs -ge 2 ]; then
    echo
    echo "Quick summary (copy-paste friendly):"
    echo "cores speedup efficiency"
    for cores in $(printf '%s\n' "${!times[@]}" | sort -n); do
        speedup=${speedups[$cores]}
        efficiency=${efficiencies[$cores]}
        printf "%5d %7.2f %9.1f\n" $cores $speedup $efficiency
    done
fi

exit 0