git push

nohup python Resnet50_5fold.py > Resnet50_5fold_output.log 2>&1 &
tail -f Resnet50_5fold_output.log
