import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.config import ISS_TELEMETRY_TOPIC, KAFKA_BOOTSTRAP_SERVERS
from sources.kafka_producer import InMemoryBroker, deserialize, resolve_backend


class ISSKafkaConsumer:
    def __init__(self, group_id: str, topic: str = ISS_TELEMETRY_TOPIC, backend: str = "auto",
                 broker: Optional[InMemoryBroker] = None, partitions: Optional[List[int]] = None):
        self.group_id, self.topic = group_id, topic
        self.backend = resolve_backend(backend)
        self._uncommitted: Dict[int, int] = {}
        if self.backend == "kafka":
            from kafka import KafkaConsumer
            self._consumer = KafkaConsumer(topic, group_id=group_id, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                                           auto_offset_reset="earliest", enable_auto_commit=False,
                                           value_deserializer=deserialize, consumer_timeout_ms=5000)
        else:
            self.broker = broker or InMemoryBroker.default()
            self.partitions = partitions if partitions is not None else list(range(self.broker.num_partitions(topic)))

    def poll(self, max_records: int = 500, timeout_ms: int = 1000) -> List[Dict[str, Any]]:
        if self.backend == "kafka":
            batches = self._consumer.poll(timeout_ms=timeout_ms, max_records=max_records)
            out = []
            for tp, records in batches.items():
                out.extend(r.value for r in records)
            return out
        recs = self.broker.fetch(self.group_id, self.topic, self.partitions, max_records)
        for p, off, _ in recs:
            self._uncommitted[p] = off + 1
        return [deserialize(v) for _, _, v in recs]

    def commit(self):
        if self.backend == "kafka":
            self._consumer.commit()
        else:
            for p, nxt in self._uncommitted.items():
                self.broker.commit(self.group_id, self.topic, p, nxt)
            self._uncommitted.clear()

    def lag(self) -> Dict[int, int]:
        return self.broker.lag(self.group_id, self.topic) if self.backend == "memory" else {}

    def close(self):
        if self.backend == "kafka":
            self._consumer.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="telemetry_processor")
    ap.add_argument("--backend", default="kafka", choices=["kafka", "memory"])
    args = ap.parse_args()
    if args.backend == "memory":
        print("The in-memory broker lives inside one process; use "
              "`python -m sources.stream_processor --source broker` to run producer + consumer together.")
        sys.exit(0)
    from sources.stream_processor import StreamPipelineProcessor
    consumer = ISSKafkaConsumer(args.group, backend="kafka")
    processor = StreamPipelineProcessor()
    total = 0
    while True:
        batch = consumer.poll(max_records=200, timeout_ms=5000)
        if not batch:
            break
        processor.process_batch(batch)
        consumer.commit()
        total += len(batch)
        print(f"consumed {total} events", end="\r")
    processor.finish()
    print(f"\nConsumer group '{args.group}' processed {total} events from Kafka.")
