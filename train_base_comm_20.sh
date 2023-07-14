#!/bin/bash

for FILE in configs/base_comm_20/* ; do 
	python main_ae.py --config $FILE
done

