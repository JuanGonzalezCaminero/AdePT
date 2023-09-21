#!/bin/bash
for i in {1..20}
do
   python3 run_scripts/run_test.py ${i} test_configurations/benchmark_varying_threads_light.json
   mkdir temp_results/${i}
   mv scaling_benchmark_results temp_results/${i}
done

python3 run_scripts/run_test.py test_configurations/benchmark_varying_threads.json