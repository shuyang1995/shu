python3 train.py --lr 0.0003 --batch-size 40 --arch resnet18 \
 	--data-root /home/boyuan/foodRecognition/data \
 	--train-list /home/boyuan/foodRecognition/data/train.txt \
 	--test-list /home/boyuan/foodRecognition/data/test.txt \
 	--model-prefix food-res18 \
 	--lr-steps 55 110 165  --epochs 220 \
 	--gpus 0 1
"# shuFood" 
