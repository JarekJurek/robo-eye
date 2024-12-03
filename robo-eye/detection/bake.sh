#!/bin/bash
### --------------- specify queue name ----------------
#BSUB -q c02516

### --------------- specify GPU request ---------------
#BSUB -gpu "num=1:mode=exclusive_process"

### --------------- specify job name ------------------
#BSUB -J newTrain

### --------------- specify number of cores -----------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### --------------- specify CPU memory requirements ---
#BSUB -R "rusage[mem=20GB]"

### --------------- specify wall-clock time (max allowed is 12:00) ---------------
#BSUB -W 12:00

### --------------- specify output and error files ---------------
#BSUB -o newTrain%J.out
#BSUB -e newTrain%J.err

### --------------- Load environment and run Python script ---------------
source /zhome/a2/c/213547/robo-eye/venv/bin/activate
python /zhome/a2/c/213547/robo-eye/robo-eye/detection/train_model2.py
