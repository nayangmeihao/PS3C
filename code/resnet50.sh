git push

nohup python Resnet50.py > Resnet50_output.log 2>&1 &
tail -f Resnet50_output.log
