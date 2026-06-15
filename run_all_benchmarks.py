import os
import sys
import json
import time

# Ensure backend path is configured
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from backend.pipeline.batch_processor import BatchProcessor

INPUT_DIR = "test_data"
ARTIFACTS_DIR = "C:/Users/DHANUNJAYA SOMIREDDY/.gemini/antigravity/brain/526bf6ed-0c27-4132-89b3-98573c19bac3"

def main():
    print("==========================================================")
    print("V4 AUTOMATED MULTI-WORKLOAD BENCHMARK RUNNER")
    print("==========================================================\n")
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    workloads = [10, 25, 50, 100]
    benchmark_results = {}
    
    for size in workloads:
        print(f"\n>>> Running Benchmark Workload Size: {size} slides...")
        processor = BatchProcessor(
            input_dir=INPUT_DIR,
            workload_size=size,
            artifacts_dir=ARTIFACTS_DIR
        )
        
        t_start = time.time()
        res = processor.execute_batch(f"v4_batch_{size}.pptx")
        t_end = time.time()
        
        # Save results for final report compilation
        benchmark_results[size] = {
            "avg_latency": res["analytics"]["average_processing_time_ms"],
            "max_ram": res["analytics"]["max_ram_usage_mb"],
            "max_vram": res["analytics"]["max_vram_usage_mb"],
            "success_rate": res["analytics"]["success_rate"],
            "reconstruction_rate": res["analytics"]["reconstruction_success_rate"],
            "editability": res["quality"]["editability_score"],
            "animation": res["quality"]["animation_ready_score"],
            "teacher": res["quality"]["teacher_ready_score"]
        }
        
        print(f"--> Done workload {size}. Avg Latency: {benchmark_results[size]['avg_latency']} ms. Max RAM: {benchmark_results[size]['max_ram']:.1f} MB.")

    # Compile the aggregated performance comparison table into the final batch_analytics_report.md
    report_path = os.path.join(ARTIFACTS_DIR, "batch_analytics_report.md")
    
    # Read existing report or write a fresh one with the dynamic table
    with open(report_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    new_lines = []
    in_table = False
    for line in lines:
        if line.startswith("## 📈 Workload Scale Performance Comparison"):
            new_lines.append(line)
            in_table = True
            continue
        if in_table:
            if line.strip() == "" or line.startswith("#") or line.startswith("##"):
                in_table = False
            else:
                # Skip the old hardcoded table rows
                continue
        new_lines.append(line)
        
    # Re-insert the dynamic table rows
    table_index = -1
    for idx, l in enumerate(new_lines):
        if l.startswith("## 📈 Workload Scale Performance Comparison"):
            table_index = idx
            break
            
    if table_index != -1:
        table_lines = [
            "\n",
            "| Workload Scale | Avg Latency / Slide | Peak RAM | Peak VRAM | Status |\n",
            "| :---: | :---: | :---: | :---: | :---: |\n"
        ]
        for size in workloads:
            res_size = benchmark_results[size]
            table_lines.append(f"| **{size} Slides** | {res_size['avg_latency']} ms | {res_size['max_ram']:.1f} MB | {res_size['max_vram']:.1f} MB | 🟢 Completed |\n")
        table_lines.append("\n")
        new_lines = new_lines[:table_index+1] + table_lines + new_lines[table_index+1:]
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
        
    # Also compile a summary json for sanity checks
    with open(os.path.join(ARTIFACTS_DIR, "benchmark_summary.json"), "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
        
    print("\n==========================================================")
    print("ALL WORKLOAD BENCHMARKS COMPLETED AND AGGREGATED")
    print(f"Reports saved in: {ARTIFACTS_DIR}")
    print("==========================================================")

if __name__ == "__main__":
    main()
