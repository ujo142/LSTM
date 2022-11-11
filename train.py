from model import LSTMModel
from dataset import LSTMDataset
import pytorch_lightning as pl


def main():
    traffic_volume =  LSTMDataset(train=True)
    model = LSTMModel()
    trainer = pl.Trainer(max_epochs=80, accelerator="cpu", log_every_n_steps=1)
    trainer.fit(model)
        
    
    
if __name__ == "__main__":
   main()