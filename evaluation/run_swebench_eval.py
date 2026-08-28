#!/usr/bin/env python3
"""
并行运行三组 predictions.jsonl 的 SWE-bench Verified 评估。

策略：96核机器，3个eval并行，每个分配 MAX_WORKERS_PER_JOB 个worker，
总并发容器数 = 3 * MAX_WORKERS_PER_JOB，建议不超过 CPU核数 * 0.75 = 72。

用法：
    python3 run_swebench_eval.py                  # 默认并行跑3组
    python3 run_swebench_eval.py --workers 24     # 每组24个worker（默认）
    python3 run_swebench_eval.py --sequential     # 顺序跑，每组72个worker
    python3 run_swebench_eval.py --jobs eval_results eval_results_top  # 只跑指定目录
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path('/path/to/eval')
DATASET_DIR = BASE_DIR / 'SWE-bench_Verified'
REPORT_BASE = BASE_DIR / 'swebench_reports'

DEFAULT_JOBS = ['eval_results', 'eval_results_top', 'eval_results_random']
TOTAL_CPU = 96
MAX_TOTAL_WORKERS = int(TOTAL_CPU * 0.75)  # 72


def run_single_eval(job_name: str, predictions_path: Path, report_dir: Path,
                    max_workers: int, run_id: str) -> dict:
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = report_dir / 'harness.log'

    cmd = [
        sys.executable, '-m', 'swebench.harness.run_evaluation',
        '--dataset_name', str(DATASET_DIR),
        '--split', 'test',
        '--predictions_path', str(predictions_path),
        '--max_workers', str(max_workers),
        '--run_id', run_id,
        '--report_dir', str(report_dir),
        '--cache_level', 'env',   # 保留env镜像，加速重复运行
        '--namespace', 'swebench',
    ]

    start = time.time()
    print(f'[{job_name}] 开始  workers={max_workers}  run_id={run_id}')
    print(f'[{job_name}] 日志 -> {log_file}')

    with open(log_file, 'w') as lf:
        lf.write(f'CMD: {" ".join(cmd)}\n\n')
        lf.flush()
        proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)

    elapsed = time.time() - start
    status = 'OK' if proc.returncode == 0 else f'FAILED(rc={proc.returncode})'
    print(f'[{job_name}] 完成  {status}  耗时={elapsed/60:.1f}min')

    # 解析结果
    result = {'job': job_name, 'run_id': run_id, 'elapsed_min': round(elapsed / 60, 1),
              'returncode': proc.returncode, 'report_dir': str(report_dir)}

    result_json = report_dir / f'{run_id}.json'
    if not result_json.exists():
        # swebench有时把结果放在子目录
        candidates = list(report_dir.rglob(f'{run_id}.json'))
        if candidates:
            result_json = candidates[0]

    if result_json.exists():
        with open(result_json) as f:
            harness = json.load(f)
        resolved = len(harness.get('resolved', []))
        total = len(harness.get('resolved', [])) + len(harness.get('unresolved', []))
        result.update({'resolved': resolved, 'total': total,
                       'resolve_rate': f'{resolved/total*100:.1f}%' if total else 'N/A'})
        print(f'[{job_name}] 结果: {resolved}/{total} resolved '
              f'({result["resolve_rate"]})')
    else:
        result['resolved'] = 'N/A'
        print(f'[{job_name}] 未找到结果JSON，请检查 {report_dir}')

    return result


def print_summary(results: list[dict]):
    print('\n' + '=' * 65)
    print('  SWE-bench Verified 评估汇总')
    print('=' * 65)
    for r in results:
        resolved = r.get('resolved', 'N/A')
        total = r.get('total', 'N/A')
        rate = r.get('resolve_rate', 'N/A')
        print(f"  {r['job']:<25}  {resolved}/{total}  ({rate})  {r['elapsed_min']}min")
    print('=' * 65)

    summary_path = REPORT_BASE / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'  汇总保存至: {summary_path}')


def main():
    parser = argparse.ArgumentParser(description='并行运行 SWE-bench Verified 评估')
    parser.add_argument('--jobs', nargs='+', default=DEFAULT_JOBS,
                        help='要评估的目录名（相对于项目根目录）')
    parser.add_argument('--workers', type=int, default=None,
                        help='每个job的max_workers（默认自动按job数均分）')
    parser.add_argument('--sequential', action='store_true',
                        help='顺序执行（每个job用全部worker）')
    parser.add_argument('--timeout', type=int, default=1800,
                        help='每个instance的超时秒数（默认1800）')
    args = parser.parse_args()

    jobs = args.jobs
    n_jobs = len(jobs)

    if args.workers:
        workers_per_job = args.workers
    elif args.sequential:
        workers_per_job = MAX_TOTAL_WORKERS
    else:
        workers_per_job = max(1, MAX_TOTAL_WORKERS // n_jobs)

    REPORT_BASE.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f'机器: {TOTAL_CPU}核  总worker上限: {MAX_TOTAL_WORKERS}')
    print(f'Jobs: {n_jobs}个  每job workers: {workers_per_job}  '
          f'总并发: {workers_per_job * (1 if args.sequential else n_jobs)}')
    print(f'模式: {"顺序" if args.sequential else "并行"}')
    print()

    # 验证predictions文件存在
    job_configs = []
    for job_name in jobs:
        pred_path = BASE_DIR / job_name / 'predictions.jsonl'
        if not pred_path.exists():
            print(f'警告: {pred_path} 不存在，跳过')
            continue
        n_lines = sum(1 for _ in open(pred_path))
        report_dir = REPORT_BASE / f'{job_name}_{run_ts}'
        run_id = f'{job_name}_{run_ts}'
        job_configs.append((job_name, pred_path, report_dir, workers_per_job, run_id))
        print(f'  {job_name}: {n_lines} predictions -> {report_dir}')

    if not job_configs:
        print('没有有效的job，退出')
        sys.exit(1)

    print()
    all_results = []

    if args.sequential:
        for cfg in job_configs:
            result = run_single_eval(*cfg)
            all_results.append(result)
    else:
        # 并行执行
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {executor.submit(run_single_eval, *cfg): cfg[0]
                       for cfg in job_configs}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as e:
                    job_name = futures[future]
                    print(f'[{job_name}] 异常: {e}')
                    all_results.append({'job': job_name, 'error': str(e)})

    print_summary(all_results)


if __name__ == '__main__':
    main()
