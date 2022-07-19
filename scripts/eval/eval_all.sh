# Evaluating MEPNet
echo "Evaluating MEPNet for componentwise/stepwise pose accuracy and MTC..."
bash scripts/eval/eval_mepnet.sh
echo "Preparing for setwise Chamfer Distance evaluation... This may take a while"
bash scripts/eval/eval_mepnet_autoreg.sh &> /dev/null
# Evaluating Direct3D
echo "Evaluating Direct3D for componentwise/stepwise pose accuracy and MTC..."
bash scripts/eval/eval_trans.sh
echo "Preparing for setwise Chamfer Distance evaluation... This may take a while"
bash scripts/eval/eval_trans_autoreg.sh &> /dev/null

# Computing Chamfer Distance
echo "Evaluating Chamfer Distance..."
PYTHONPATH=. python scripts/eval/eval_cd.py
