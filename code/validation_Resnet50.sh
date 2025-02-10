git push

nohup python validation_Resnet50.py > validation_Resnet50_output.log &
tail -f validation_Resnet50_output.log

