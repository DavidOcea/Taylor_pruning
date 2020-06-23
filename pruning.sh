




20200516 pruning,1），log：0516，用第一次的: 
nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0516 --dataset=mult_5T \
--lr=0.01 --lr-decay-every=10 --momentum=0.9 --epochs=60  --pruning=False --seed=0 --model=purn_20200411_5T_2b \
--load_model=./Tools/models/purn_20200411_5T_2b.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0001 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True --fineturn_model=True \
--pruning_config=./configs/mult_5T_prune72_406.json >./logs/20200516_mult5T_pruning.log 2>&1 &
20200515 pruning,1）单独清洗了persoon数据，看能否比14号好一点，14号的改进了部分问题log：0515 2）该部分训到一般被我停了，因为other太低
原因疑似重新生成txt重复比例不同，所以重巡，结果发现还不如第一次，所以停了用第一次数据接着训;log：0515—_2，用wm_train2: 
nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0515_2 --dataset=mult_5T \
--lr=0.01 --lr-decay-every=10 --momentum=0.9 --epochs=60  --pruning=False --seed=0 --model=purn_20200411_5T_2b \
--load_model=./Tools/models/purn_20200411_5T_2b.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0001 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True --fineturn_model=True \
--pruning_config=./configs/mult_5T_prune72_406.json >./logs/20200515_mult5T_pruning_2.log 2>&1 &
20200514 pruning,更新非机动车车型后训练: nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0514 --dataset=mult_5T \
--lr=0.01 --lr-decay-every=10 --momentum=0.9 --epochs=60  --pruning=False --seed=0 --model=purn_20200411_5T_2b \
--load_model=./Tools/models/purn_20200411_5T_2b.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0001 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True --fineturn_model=True \
--pruning_config=./configs/mult_5T_prune72_406.json >./logs/20200514_mult5T_pruning.log 2>&1 &
20200510 pruning,更新非机动车车型后训练: nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0510 --dataset=mult_5T \
--lr=0.01 --lr-decay-every=10 --momentum=0.9 --epochs=60  --pruning=True --seed=0 --model=purn_20200411_5T_2b \
--load_model=./Tools/models/purn_20200411_5T_2b.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True --fineturn_model=True \
--pruning_config=./configs/mult_5T_prune72_406.json >./logs/20200510_mult5T_pruning.log 2>&1 &
20200418 pruning,更新非机动车车型后训练 ngpu设为4了: nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0418 --dataset=mult_5T \
--lr=0.01 --lr-decay-every=10 --momentum=0.9 --epochs=60  --pruning=True --seed=0 --model=purn_20200411_5T_2b \
--load_model=./Tools/models/purn_20200411_5T_2b.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True --fineturn_model=True \
--pruning_config=./configs/mult_5T_prune72_406.json >./logs/20200429_mult5T_pruning.log 2>&1 &
20200418 pruning: nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0418 --dataset=mult_5T \
--lr=0.01 --lr-decay-every=10 --momentum=0.9 --epochs=60  --pruning=True --seed=0 --model=purn_20200411_5T_2b \
--load_model=./Tools/models/purn_20200411_5T_2b.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True --fineturn_model=True \
--pruning_config=./configs/mult_5T_prune72_406.json >./logs/20200412_mult5T_pruning.log 2>&1 &
20200415 pruning:nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0415 --dataset=mult_5T \
--lr=0.01 --lr-decay-every=10 --momentum=0.9 --epochs=60  --pruning=True --seed=0 --model=mult_prun_5T_normal \
--load_model=./models/pretrained/nma_20200223_4T.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True --fineturn_model=True \
--pruning_config=./configs/mult_5T_prune72_406.json >./logs/20200418_mult5T.log 2>&1 &
20200414 pruning:nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0414 --dataset=mult_5T \
--lr=0.001 --lr-decay-every=10 --momentum=0.9 --epochs=20  --pruning=True --seed=0 --model=mult_prun_5T_normal \
--load_model=./models/pretrained/nma_20200224_5T.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True \
--pruning_config=./configs/mult_5T_prune72_406.json >./logs/20200414_mult5T.log 2>&1 &
20200411 pruning:nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0412 --dataset=mult_5T \
--lr=0.001 --lr-decay-every=10 --momentum=0.9 --epochs=20  --pruning=True --seed=0 --model=purn_20200411_5T_2b \
--load_model=./Tools/models/purn_20200411_5T_2b.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True \
--pruning_config=./configs/mult_5T_prune72_406.json >./logs/20200412_mult5T_pruning.log 2>&1 &
20200411 pruning:nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0411_t --dataset=mult_5T \
--lr=0.001 --lr-decay-every=10 --momentum=0.9 --epochs=20  --pruning=True --seed=0 --model=multprun_gate5_gpu_0316_1 \
--load_model=./models/pretrained/nma_20200319_5T.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True \
--pruning_config=./configs/mult_5T_prune72_406.json >./logs/20200411_mult5T_pruning_t.log 2>&1 &
20200410 pruning:nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0410 --dataset=mult_5T \
--lr=0.001 --lr-decay-every=10 --momentum=0.9 --epochs=20  --pruning=True --seed=0 --model=multprun_gate5_gpu_0316_1 \
--load_model=./models/pretrained/nma_20200319_5T.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True \
--pruning_config=./configs/mult_5T_prune72_406.json >./logs/20200410_mult5T_pruning.log 2>&1 &
20200408_2 pruning:nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0408_2 --dataset=mult_5T \
--lr=0.001 --lr-decay-every=10 --momentum=0.9 --epochs=20  --pruning=True --seed=0 --model=multprun_gate5_gpu_0316_1 \
--load_model=./models/pretrained/nma_20200319_5T.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True \
--pruning_config=./configs/mult_5T_prune72_406.json >./logs/20200408_mult5T_pruning_2.log 2>&1 &
20200408 pruning:nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0408 --dataset=mult_5T \
--lr=0.001 --lr-decay-every=10 --momentum=0.9 --epochs=20  --pruning=True --seed=0 --model=multprun_gate5_gpu_0316_1 \
--load_model=./models/pretrained/nma_20200319_5T.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True \
--pruning_config=./configs/mult_5T_prune72_406.json >./logs/20200408_mult5T_pruning.log 2>&1 &
20200407 pruning:nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0407 --dataset=mult_5T \
--lr=0.001 --lr-decay-every=10 --momentum=0.9 --epochs=20  --pruning=True --seed=0 --model=multprun_gate5_gpu_0316_1 \
--load_model=./models/pretrained/nma_20200319_5T.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True \
--pruning_config=./configs/mult_5T_prune72_406.json >./logs/20200407_mult5T_pruning.log 2>&1 &
20200406 pruning:nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0406 --dataset=mult_5T \
--lr=0.001 --lr-decay-every=10 --momentum=0.9 --epochs=20  --pruning=True --seed=0 --model=multprun_gate5_gpu_0316_1 \
--load_model=./models/pretrained/nma_20200319_5T.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 \
--tensorboard=False --pruning-method=22 --data='' --no_grad_clip=True \
--pruning_config=./configs/mult_5T_prune72_404.json >./logs/20200406_mult5T_pruning.log 2>&1 &
20200404 pruning:nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0404 --dataset=mult_5T \
--lr=0.001 --lr-decay-every=10 --momentum=0.9 --epochs=30  --pruning=True --seed=0 --model=multnas5_gpu_0316 \
--load_model=./models/pretrained/nma_20200319_5T.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True \
--pruning_config=./configs/mult_5T_prune72_404.json >./logs/20200404_mult5T_pruning.log 2>&1 &
20200403 pruning:nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0403 --dataset=mult_5T \
--lr=0.001 --lr-decay-every=10 --momentum=0.9 --epochs=30  --pruning=True --seed=0 --model=multnas5_gpu_0316 \
--load_model=./models/pretrained/nma_20200319_5T.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True \
--pruning_config=./configs/mult_5T_prune72.json >./logs/20200403_mult5T_pruning.log 2>&1 &
20200401 pruning:nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72_0401 --dataset=mult_5T \
--lr=0.001 --lr-decay-every=10 --momentum=0.9 --epochs=40  --pruning=True --seed=0 --model=multnas5_gpu_0316 \
--load_model=./models/pretrained/nma_20200319_5T.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True \
--pruning_config=./configs/mult_5T_prune72.json >./logs/20200401_mult5T_pruning.log 2>&1 &
20200331 pruning:nohup python -u main.py --name=runs/mult_5T/mult_5T_prune72 --dataset=mult_5T \
--lr=0.001 --lr-decay-every=10 --momentum=0.9 --epochs=40  --pruning=True --seed=0 --model=multnas5_gpu_0316 \
--load_model=./models/pretrained/nma_20200319_5T.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True \
--pruning_config=./configs/mult_5T_prune72.json >./logs/20200331_mult5T_pruning.log 2>&1 &



python main.py --name=runs/mult_5T/mult_5T_prune72_0503 --dataset=mult_5T \
--lr=0.01 --lr-decay-every=10 --momentum=0.9 --epochs=60  --pruning=True --seed=0 --model=purn_20200411_5T_2b \
--load_model=./Tools/models/purn_20200411_5T_2b.pth.tar --mgpu=True --group_wd_coeff=1e-8 --wd=0.0 \
--tensorboard=True --pruning-method=22 --data='' --no_grad_clip=True --fineturn_model=True \
--pruning_config=./configs/mult_5T_prune72_406.json