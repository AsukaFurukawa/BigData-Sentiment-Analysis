from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

message = {"test": "Kafka connection working!"}
producer.send("reddit-raw-data", message)
producer.flush()

print("✅ Test message sent to kafka topic: reddit-raw-data")
