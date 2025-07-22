pm2 start python3 --name nova-miner -- \
  neurons/miner.py \
  --wallet.name nicholas-miner1 \
  --wallet.hotkey nicholas-miner1-hot1 \
  --wallet.path /workspace/.bittensor/wallets \
  --logging.info