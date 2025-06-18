#!/bin/bash
echo "启动 iRoopDeepFaceCam..."
source venv/bin/activate
python3 run.py --execution-provider coreml --max-memory 4
