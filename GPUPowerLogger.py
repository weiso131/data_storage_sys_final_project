import threading
import subprocess
import time

class GPUPowerLogger(threading.Thread):
    def __init__(self, gpu_index=0, interval=0.01):
        super().__init__()
        self.gpu_index = gpu_index
        self.interval = interval
        self.running = False
        self.records = []  # (timestamp, power)

    def run(self):
        self.running = True
        while self.running:
            ts = time.perf_counter()
            try:
                # 使用 nvidia-smi 查詢 GPU 功耗
                result = subprocess.run(
                    ["nvidia-smi",
                     f"--query-gpu=power.draw",
                     "--format=csv,noheader,nounits",
                     "-i", str(self.gpu_index)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                power = float(result.stdout.strip())  # Watt
                self.records.append((ts, power))
            except Exception as e:
                print("Error reading GPU power:", e)
            time.sleep(self.interval)

    def stop(self):
        self.running = False

    def get_average_power(self, records):
        """
        records: list of tuples [(start_time, end_time), ...]
        """
        output = 0
        cnt = 0
        for start_time, end_time in records:
            powers = [p for t, p in self.records if start_time <= t <= end_time]
            if powers:
                output += sum(powers)
                cnt += len(powers)
        return output / cnt if cnt > 0 else 0
