echo '#!/bin/bash
# Create reddit topic if it doesn't exist
kafka-topics --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 3 --topic reddit-raw-data --if-not-exists' > kafka-setup/kafka-topic.sh