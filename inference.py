from model import LSTMModel
from dataset import LSTMDataset
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pickle import load
import matplotlib.pyplot as plt


def main():
    PATH = "/Users/ben/python_projects/LSTM_weather/lightning_logs/version_44/checkpoints/epoch=19-step=2480.ckpt"
    
    model = LSTMModel()
    trained_traffic_volume_TrafficVolumePrediction = model.load_from_checkpoint(PATH)
    trained_traffic_volume_TrafficVolumePrediction.eval()

  
    traffic_volume_test_dataset =  LSTMDataset(test=True)
    traffic_volume_test_dataloader = torch.utils.data.DataLoader(traffic_volume_test_dataset, batch_size=16)
    predicted_result, actual_result = [], []
    for i,j in traffic_volume_test_dataloader:
        print(i.shape,j.shape)
    

    for i, (features,targets) in enumerate(traffic_volume_test_dataloader):
        result = trained_traffic_volume_TrafficVolumePrediction(torch.tensor(features, device="cpu"))
        predicted_result.extend(result.view(-1).tolist())
        actual_result.extend(targets.view(-1).tolist())
        
    scaler = load(open('scaler_sunspots.pkl', 'rb'))
    actual_predicted_df = pd.DataFrame(data={"actual":actual_result, "predicted": predicted_result})
    inverse_transformed_values = scaler.inverse_transform(actual_predicted_df)
    actual_predicted_df["actual"] = inverse_transformed_values[:,[0]]
    actual_predicted_df["predicted"] = inverse_transformed_values[:,[1]]
    plt.plot(actual_predicted_df["actual"],'b')
    plt.plot(actual_predicted_df["predicted"],'r')
    plt.show()
    
    
    
    
if __name__ == "__main__":
    main()