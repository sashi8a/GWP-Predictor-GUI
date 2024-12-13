import tkinter as tk
from tkinter import messagebox
import pandas as pd
from padelpy import from_smiles
import torch
from model import CustomNN

def load_best_model(model_path, input_dim, output_dim):
    # Create an instance of the model
    model = CustomNN(input_dim, output_dim)
    
    # Load the saved state dictionary
    model.load_state_dict(torch.load(model_path))
    
    # Set the model to evaluation mode
    model.eval()
    
    return model


input_dim = 105  
output_dim = 1   
best_model_path = "best_custom_nn_model.pth"

best_model = load_best_model(best_model_path, input_dim, output_dim)


# Function to compute molecular descriptors from SMILES string using PadelPy
def compute_descriptors(smiles):
    try:
        s = str(smiles)
        descriptors = from_smiles(s)
        if not descriptors:
            raise ValueError("Invalid SMILES string or failed to compute descriptors")
        # Convert descriptor dictionary to a DataFrame row
        descriptors_df = pd.DataFrame([descriptors])
        filtered_df = descriptors_df[['GATS2m','ETA_Epsilon_5','MLFER_BH','CIC2','GATS5p','AATSC4s',
        'MATS5p','VE1_Dzs','MATS2p','AATSC1m','bpol','AATSC2p','MIC1','MATS3e','CIC0','ATSC6m',
        'ATSC2m','MDEC-13','GATS1i','ETA_EtaP_F','SM1_Dzs','ATSC4s','SpMax4_Bhv','AMR','AVP-3',
        'nBr','GATS4c','SM1_DzZ','SpAbs_DzZ','AVP-0','ATS1s','ETA_EtaP_L','AATSC4p','SpMax4_Bhm',
        'CIC3','minHBint4','MATS2e','SpMin5_Bhs','SpMin6_Bhs','hmin','MLFER_E','GATS3s','MATS4c',
        'VR1_Dzv','MATS4i','MATS6i','nHeteroRing','AATSC4m','SpMin3_Bhe','ATSC4m','AATSC0m',
        'VR1_Dzs','ETA_Eta_F_L','ATSC6i','SpMax1_Bhm','MATS6c','MATS2v','MATS2c','AATSC2v',
        'ATSC0e','SpMax8_Bhs','GATS3e','nBondsD','ATS0p','MATS4m','ETA_Eta_F','MATS4s','GATS5v',
        'ATSC7s','GATS2i','ATSC1m','MATS2i','GATS1v','ATSC7e','MATS2m','SwHBa','AATS2m',
        'SpMin7_Bhe','AATSC3m','minaasC','maxssCH2','GATS3m','MATS2s','VE3_Dze','ALogP','SpMin6_Bhe',
        'ATSC3s','MATS5c','ETA_dPsi_A','ATSC6v','ATSC0p','GATS1e','MATS3c','ATSC4v','GATS4s','VE3_Dzm',
        'ATSC5c','MATS4e','BCUTw-1l','GATS5i','AATSC5v','ATSC4i','SpMin8_Bhs','MATS1e','ATSC7v']]
        print(filtered_df.head())
        filtered_df.to_csv(s+'_descriptors.csv', index=False)
        return filtered_df.values
    except Exception as e:
        messagebox.showerror("Error", f"Failed to compute descriptors: {e}")
        return None



#Function to make prediction based on SMILES string
def predict_gwp(smiles_entry):

    try:
        s = str(smiles_entry)
        descriptors = from_smiles(s)
        if not descriptors:
            raise ValueError("Invalid SMILES string or failed to compute descriptors")
        # Convert descriptor dictionary to a DataFrame row
        descriptors_df = pd.DataFrame([descriptors])
        filtered_df = descriptors_df[['GATS2m','ETA_Epsilon_5','MLFER_BH','CIC2','GATS5p','AATSC4s',
        'MATS5p','VE1_Dzs','MATS2p','AATSC1m','bpol','AATSC2p','MIC1','MATS3e','CIC0','ATSC6m',
        'ATSC2m','MDEC-13','GATS1i','ETA_EtaP_F','SM1_Dzs','ATSC4s','SpMax4_Bhv','AMR','AVP-3',
        'nBr','GATS4c','SM1_DzZ','SpAbs_DzZ','AVP-0','ATS1s','ETA_EtaP_L','AATSC4p','SpMax4_Bhm',
        'CIC3','minHBint4','MATS2e','SpMin5_Bhs','SpMin6_Bhs','hmin','MLFER_E','GATS3s','MATS4c',
        'VR1_Dzv','MATS4i','MATS6i','nHeteroRing','AATSC4m','SpMin3_Bhe','ATSC4m','AATSC0m',
        'VR1_Dzs','ETA_Eta_F_L','ATSC6i','SpMax1_Bhm','MATS6c','MATS2v','MATS2c','AATSC2v',
        'ATSC0e','SpMax8_Bhs','GATS3e','nBondsD','ATS0p','MATS4m','ETA_Eta_F','MATS4s','GATS5v',
        'ATSC7s','GATS2i','ATSC1m','MATS2i','GATS1v','ATSC7e','MATS2m','SwHBa','AATS2m',
        'SpMin7_Bhe','AATSC3m','minaasC','maxssCH2','GATS3m','MATS2s','VE3_Dze','ALogP','SpMin6_Bhe',
        'ATSC3s','MATS5c','ETA_dPsi_A','ATSC6v','ATSC0p','GATS1e','MATS3c','ATSC4v','GATS4s','VE3_Dzm',
        'ATSC5c','MATS4e','BCUTw-1l','GATS5i','AATSC5v','ATSC4i','SpMin8_Bhs','MATS1e','ATSC7v']]

        # Ensure all columns are numeric
        filtered_df = filtered_df.apply(pd.to_numeric, errors='coerce')
        if filtered_df.isnull().any().any():
            raise ValueError("Descriptors contain non-numeric values or NaNs")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to compute descriptors: {e}")
        return None
    
    if filtered_df is not None:
        try:
            # Convert DataFrame to tensor
            input_tensor = torch.tensor(filtered_df.values, dtype=torch.float32)
            print('passed')
            device = next(best_model.parameters()).device
            print('passed')

            input_tensor = input_tensor.to(device)  # Ensure it uses the same device as the model
            print('passed')

            # Perform prediction using the NN
            best_model.eval()
            with torch.no_grad():
                prediction = best_model(input_tensor).cpu().numpy()

            print('passed')
            prediction = prediction[0][0]  # Extract the scalar value from the output
            print('passed')

            result_label.config(text=f"Predicted GWP: {prediction:.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")





# Create the Tkinter GUI
root = tk.Tk()
root.title("GWP Prediction from SMILES")
root.geometry("400x250")

# SMILES input label and entry
smiles_label = tk.Label(root, text="Enter SMILES string:")
smiles_label.pack(pady=10)

smiles_entry = tk.Entry(root, width=50)
smiles_entry.pack(pady=5)

# Retrieve Descriptor button
descriptor_button = tk.Button(
    root, 
    text="Retrieve SMILES Descriptor Data", 
    command=lambda: compute_descriptors(smiles_entry.get())
)
descriptor_button.pack(pady=10)

# Predict button
predict_button = tk.Button(
    root, 
    text="Predict GWP",
    command=lambda: predict_gwp(smiles_entry.get())
)
predict_button.pack(pady=10)

# Result label
result_label = tk.Label(root, text="Predicted GWP: ")
result_label.pack(pady=20)








root.mainloop()
