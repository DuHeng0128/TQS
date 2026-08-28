#!/usr/bin/env python3
"""
从镜像加速器并行预拉取 SWE-bench Verified 500个 instance 镜像，
然后retag为swebench官方名称，让swebench harness直接复用本地缓存。

镜像名: swebench/sweb.eval.x86_64.<instance_id_lower>:latest
加速器: docker.1ms.run

策略:
- 16 路并行 pull (避免触发加速器限流)
- pull 后立刻 retag 为官方名 + 删除带前缀的tag (节省命名空间但保留image)
- 失败的镜像记录到 failed.txt, 可重跑
- 进度通过 state.json 持久化, 中断后能续跑
"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BASE = Path('/path/to/eval')
DATASET = BASE / 'SWE-bench_Verified/data/test-00000-of-00001.parquet'
MIRROR = 'docker.1ms.run'
OFFICIAL_NS = 'swebench'

STATE_DIR = BASE / 'swebench_reports' / '_image_cache'
STATE_FILE = STATE_DIR / 'state.json'
LOG_FILE = STATE_DIR / 'pull.log'


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {'done': [], 'failed': {}}


def save_state(state):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def image_name_for(instance_id: str) -> str:
    # swebench 把 instance_id 中的 __ 替换为 _1776_ (见 test_spec.py:110)
    encoded = instance_id.lower().replace('__', '_1776_')
    return f'sweb.eval.x86_64.{encoded}:latest'


def has_local_image(name_with_ns: str) -> bool:
    r = subprocess.run(['docker', 'image', 'inspect', name_with_ns],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0


def pull_one(instance_id: str, retries: int = 2) -> tuple[str, bool, str]:
    img = image_name_for(instance_id)
    official = f'{OFFICIAL_NS}/{img}'
    mirror = f'{MIRROR}/{OFFICIAL_NS}/{img}'

    if has_local_image(official):
        return instance_id, True, 'cached'

    last_err = ''
    for attempt in range(retries + 1):
        # 1. pull 镜像
        p = subprocess.run(['docker', 'pull', mirror],
                           capture_output=True, text=True, timeout=900)
        if p.returncode != 0:
            last_err = (p.stderr or p.stdout)[-300:].strip()
            time.sleep(5 * (attempt + 1))
            continue

        # 2. retag 成官方名
        t = subprocess.run(['docker', 'tag', mirror, official],
                           capture_output=True, text=True)
        if t.returncode != 0:
            last_err = f'tag failed: {t.stderr.strip()}'
            continue

        # 3. 删除带 mirror 前缀的引用 (释放重复tag, 不删image)
        subprocess.run(['docker', 'rmi', mirror],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return instance_id, True, 'pulled'

    return instance_id, False, last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16,
                        help='并行pull数 (默认16, 避免限流)')
    parser.add_argument('--instance-ids', nargs='+', default=None,
                        help='指定要拉的instance_id, 不指定则拉SWE-bench Verified全部500个')
    parser.add_argument('--retry-failed', action='store_true',
                        help='只重试上次失败的')
    parser.add_argument('--print-only', action='store_true',
                        help='只打印计划, 不实际拉取')
    args = parser.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()

    # 决定要拉的列表
    if args.instance_ids:
        targets = list(args.instance_ids)
    elif args.retry_failed:
        targets = list(state.get('failed', {}).keys())
    else:
        df = pd.read_parquet(DATASET)
        targets = df['instance_id'].tolist()

    # 过滤已完成的
    done_set = set(state.get('done', []))
    pending = [iid for iid in targets if iid not in done_set]

    print(f'目标镜像数: {len(targets)}')
    print(f'已完成    : {len(done_set & set(targets))}')
    print(f'待处理    : {len(pending)}')
    print(f'并发      : {args.workers}')
    print(f'加速器    : {MIRROR}')

    if args.print_only or not pending:
        return

    log = open(LOG_FILE, 'a')
    log.write(f'\n=== START {time.strftime("%Y-%m-%d %H:%M:%S")} {len(pending)} images ===\n')
    log.flush()

    start = time.time()
    success = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(pull_one, iid): iid for iid in pending}
        for i, fut in enumerate(as_completed(futures), 1):
            iid, ok, msg = fut.result()
            elapsed = time.time() - start
            rate = i / elapsed if elapsed else 0
            eta = (len(pending) - i) / rate if rate else 0

            tag = '[OK]' if ok else '[FAIL]'
            line = (f'{tag} ({i}/{len(pending)}) {iid:<55} {msg:<20} '
                    f'speed={rate:.2f}/s eta={eta/60:.1f}min')
            print(line)
            log.write(line + '\n')
            log.flush()

            if ok:
                success += 1
                if iid not in done_set:
                    state['done'].append(iid)
                state.get('failed', {}).pop(iid, None)
            else:
                failed += 1
                state.setdefault('failed', {})[iid] = msg

            # 每20个保存一次state
            if i % 20 == 0:
                save_state(state)

    save_state(state)
    log.close()

    total_min = (time.time() - start) / 60
    print(f'\n完成: {success} 成功, {failed} 失败, 耗时 {total_min:.1f}min')
    if failed:
        print(f'失败列表已记录在 {STATE_FILE}')
        print(f'重跑失败的: python3 {sys.argv[0]} --retry-failed')


if __name__ == '__main__':
    main()
