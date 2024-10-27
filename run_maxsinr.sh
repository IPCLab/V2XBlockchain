#!/bin/bash

mkdir -p output_random_seed
# Define the arrays
seeds=(20 42 67 168 1234)
method="maxsinr"
numbers=(100 200 300 400 500)

# Define the number of parallel jobs (threads)
NUM_THREADS=16

# Create a function to run the Python command
run_command() {
    seed=$1
    method=$2
    number=$3
    echo "MAAC_a_MAX_SINR.py "$seed" "$method" "data_test_area_${number}.csv" > "./output_random_seed/${method}_650_0501_1020_${number}veh_${seed}.txt""
    python -u MAAC_a_MAX_SINR.py "$seed" "$method" "data_test_area_${number}.csv" > "./output_random_seed/${method}_650_0501_1020_${number}veh_${seed}.txt"
}

# Run commands in parallel
for seed in "${seeds[@]}"; do
    for number in "${numbers[@]}"; do
        while [ "$job_count" -ge "$NUM_THREADS" ]; do
            # Wait for any job to finish before starting a new one
            wait -n
            job_count=$((job_count - 1))
        done

        run_command "$seed" "$method" "$number" &
        job_count=$((job_count + 1))
    done
done

# Wait for any remaining jobs to finish
wait

echo "save result into output_random_seed/output.csv"
python get_txt_info.py output_random_seed/

echo "Task completed."



