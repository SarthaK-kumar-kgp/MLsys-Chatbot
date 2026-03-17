import click
import json
import os
import sys
import asyncio
import aiohttp
import time
import numpy as np
from tqdm.asyncio import tqdm
import traceback

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

class AsyncEngineClient:
    def __init__(self, base_url="http://localhost:8000", timeout=30):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def check_health(self):
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            max_retries = 60
            for i in range(max_retries):
                try:
                    async with session.get(f"{self.base_url}/ready") as resp:
                        if resp.status == 200:
                            return True
                        print(f"Waiting for engine... (HTTP {resp.status})")
                except Exception as e:
                    print(f"Waiting for engine... ({e})")
                
                await asyncio.sleep(5)
            return False

    async def run_chat_completion(self, session, chat_req):
        start = time.time()
        async with session.post(f"{self.base_url}/v1/chat/completions", json=chat_req, timeout=self.timeout) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"HTTP {resp.status}: {text}")
            try:
                data = await resp.json()
            except Exception as e:
                text = await resp.text()
                raise Exception(f"JSON Parse Error: {e} | Body: {text[:200]}")
            end = time.time()
            return data, end - start

@click.command()
@click.option('--url', default='http://localhost:8000', help='URL of the engine')
@click.option('--data', required=True, type=click.Path(exists=True), help='Path to the performance prompts JSONL file')
@click.option('--concurrency', default=100, help='Concurrency level')
@click.option('--timeout', default=30, type=int, help='Request timeout in seconds')
def main(url, data, concurrency, timeout):
    """Benchmark suite for Track 2: Customer Chat."""
    if not data.endswith('.jsonl') and not data.endswith('.txt'): # Allowing .txt if it contains JSONL
        pass 
    # Strict JSONL check
    print(f"Connecting to engine at {url} using data {data} (concurrency={concurrency}, timeout={timeout}s)...")
    asyncio.run(run_performance(url, data, concurrency, timeout))

async def run_performance(url, data_path, concurrency, timeout):
    client = AsyncEngineClient(url, timeout=timeout)
    if not await client.check_health():
        print("Error: Engine not healthy.")
        sys.exit(1)

    print("=== Track 2: Performance Check ===")
    
    prompts = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
                prompts.append(data['instruction'])
            except json.JSONDecodeError:
                print(f"Error: Performance data must be JSONL. Failed to parse line: {line[:50]}...")
                sys.exit(1)
        
    if not prompts:
        print(f"Error: No prompts found in {data_path}")
        sys.exit(1)

    # Generate workload
    work_items = prompts
    
    results = []
    sem = asyncio.Semaphore(concurrency)
    
    async def worker(prompt):
        async with sem:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "max_tokens": 256 # Higher for performance test
                }
                try:
                    response, latency = await client.run_chat_completion(session, payload)
                    if 'output' not in response:
                        print(f"\\n[FAIL] Prompt: {prompt[:30]}... Missing 'output' in response: {response}")
                        return {"success": False, "latency": latency, "perplexity": None}
                    
                    logprobs = response['logprobs']
                    avg_logprob = sum(logprobs) / len(logprobs)
                    perplexity = np.exp(-avg_logprob)

                    return {"success": True, "latency": latency, "perplexity": perplexity}
                except Exception as e:
                    print(f"\\n[ERROR] Prompt: {prompt[:30]}... Exception: {e}")
                    traceback.print_exc()
                    return {"success": False, "latency": 0, "perplexity": None}

    start_time = time.time()
    tasks_to_run = [worker(p) for p in work_items]
    
    print(f"Executing {len(work_items)} requests with concurrency {concurrency}...")
    for f in tqdm(asyncio.as_completed(tasks_to_run), total=len(tasks_to_run)):
        results.append(await f)
        
    total_time = time.time() - start_time
    passed = sum(1 for r in results if r['success'])
    failed = len(results) - passed
    latencies = [r['latency'] for r in results if r['success']]
    perplexities = [r['perplexity'] for r in results if r['success']]
    
    print(f"\nPerformance Metrics:")
    print(f"  Throughput: {len(results) / total_time:.2f} req/s")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    if latencies:
        print(f"  Avg Latency: {np.mean(latencies):.4f}s")
        print(f"  P50 Latency: {np.percentile(latencies, 50):.4f}s")
        print(f"  P99 Latency: {np.percentile(latencies, 99):.4f}s")
    if perplexities:
        print(f"  Avg Perplexity: {np.mean(perplexities):.4f}")

if __name__ == '__main__':
    main()
