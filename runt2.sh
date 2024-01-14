#!/bin/bash
for i in {1..200}
do
  pytest tests/test_bug.py --count=3 -v  -s > outs/out$i.log
done
