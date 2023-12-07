#!/bin/bash
for i in {1..100}
do
  pytest tests/test_bug.py --count=3 -v | tee  >(tail -n 1 >> to1.log)
done
