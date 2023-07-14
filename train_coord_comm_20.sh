#!/bin/bash

for FILE in configs/coord_comm_20/* ; do 
	python main_ae.py --config $FILE
done

