import argparse
import json
import sys
import threading
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sources.config import (ISS_TELEMETRY_TOPIC, KAFKA_BOOTSTRAP_SERVERS,
                            KAFKA_NUM_PARTITIONS, RAW_DATA_PATH)


def partition_for(key: str, num_partitions: int) -> int:
    return zlib.crc32(key.encode()) % num_partitions


def serialize(event: Dict[str, Any]) -> bytes:
    return json.dumps(event, default=str).encode("utf-8")


def deserialize(raw: bytes) -> Dict[str, Any]:
    return json.loads(raw.decode("utf-8"))


class InMemoryBroker:
    _default: Optional["InMemoryBroker"] = None

    def __init__(self):
        self._topics: Dict[str, List[List[Tuple[int, str, bytes]]]] = {}
        self._offsets: Dict[Tuple[str, str, int], int] = {}
        self._lock = threading.Lock()

    @classmethod
    def default(cls) -> "InMemoryBroker":
        if cls._default is None:
            cls._default = cls()
        return cls._default

    def create_topic(self, topic: str, num_partitions: int = KAFKA_NUM_PARTITIONS):
        with self._lock:
            self._topics.setdefault(topic, [[] for _ in range(num_partitions)])

    def num_partitions(self, topic: str) -> int:
        return len(self._topics[topic])

    def produce(self, topic: str, key: str, value: bytes) -> Tuple[int, int]:
        with self._lock:
            parts = self._topics[topic]
            p = partition_for(key, len(parts))
            offset = len(parts[p])
            parts[p].append((offset, key, value))
            return p, offset

    def fetch(self, group: str, topic: str, partitions: List[int], max_records: int) -> List[Tuple[int, int, bytes]]:
        out = []
        with self._lock:
            for p in partitions:
                pos = self._offsets.get((group, topic, p), 0)
                log = self._topics[topic][p]
                for off, _, val in log[pos: pos + max_records - len(out)]:
                    out.append((p, off, val))
                if len(out) >= max_records:
                    break
        return out

    def commit(self, group: str, topic: str, partition: int, next_offset: int):
        with self._lock:
            self._offsets[(group, topic, partition)] = next_offset

    def lag(self, group: str, topic: str) -> Dict[int, int]:
        return {p: len(log) - self._offsets.get((group, topic, p), 0) for p, log in enumerate(self._topics[topic])}

    def partition_sizes(self, topic: str) -> List[int]:
        return [len(p) for p in self._topics[topic]]


def kafka_available(bootstrap: str = KAFKA_BOOTSTRAP_SERVERS) -> bool:
    try:
        from kafka import KafkaAdminClient
    except ImportError:
        return False
    try:
        from kafka import KafkaAdminClient
        KafkaAdminClient(bootstrap_servers=bootstrap, request_timeout_ms=3000).close()
        return True
    except Exception:
        return False


def resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "kafka" if kafka_available() else "memory"
    return backend


class ISSKafkaProducer:
    def __init__(self, topic: str = ISS_TELEMETRY_TOPIC, backend: str = "auto",
                 broker: Optional[InMemoryBroker] = None, num_partitions: int = KAFKA_NUM_PARTITIONS):
        self.topic = topic
        self.backend = resolve_backend(backend)
        self.stats = {"sent": 0, "dropped": 0, "per_partition": {}}
        if self.backend == "kafka":
            from kafka import KafkaProducer
            from kafka.admin import KafkaAdminClient, NewTopic
            from kafka.errors import TopicAlreadyExistsError
            admin = KafkaAdminClient(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
            try:
                admin.create_topics([NewTopic(topic, num_partitions=num_partitions, replication_factor=1)])
            except TopicAlreadyExistsError:
                pass
            admin.close()
            self._producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, acks="all", linger_ms=5,
                                           key_serializer=str.encode, value_serializer=serialize)
        else:
            self.broker = broker or InMemoryBroker.default()
            self.broker.create_topic(topic, num_partitions)

    def publish_event(self, event: Dict[str, Any]) -> bool:
        if not isinstance(event, dict) or "event_id" not in event:
            self.stats["dropped"] += 1
            return False
        key = str(event.get("satellite_id", "unknown"))
        if self.backend == "kafka":
            self._producer.send(self.topic, key=key, value=event)
            p = partition_for(key, KAFKA_NUM_PARTITIONS)
        else:
            p, _ = self.broker.produce(self.topic, key, serialize(event))
        self.stats["sent"] += 1
        self.stats["per_partition"][p] = self.stats["per_partition"].get(p, 0) + 1
        return True

    def flush(self):
        if self.backend == "kafka":
            self._producer.flush()

    def stream_dataset(self, generator, max_records: Optional[int] = None) -> int:
        n = 0
        for event in generator.generate_events():
            n += self.publish_event(event)
            if max_records and n >= max_records:
                break
        self.flush()
        return n


ISSKafkaProducerSimulator = ISSKafkaProducer


if __name__ == "__main__":
    from sources.stream_generator import ISSStreamGenerator
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto", choices=["auto", "memory", "kafka"])
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--faults", action="store_true", help="inject labelled faults")
    args = ap.parse_args()
    producer = ISSKafkaProducer(backend=args.backend)
    n = producer.stream_dataset(ISSStreamGenerator(RAW_DATA_PATH, inject_faults=args.faults), args.max_records)
    print(f"Backend: {producer.backend}. Published {n} events to topic '{producer.topic}'. Stats: {producer.stats}")
    if producer.backend == "memory":
        print("Partition sizes:", producer.broker.partition_sizes(producer.topic))
        print("(The in-memory topic lives only in this process; stream_processor.py --source broker "
              "runs producer and consumers together.)")