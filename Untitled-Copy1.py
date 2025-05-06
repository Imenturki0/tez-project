{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6f215161-7c3b-4a47-b38a-bc2dd01b138b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: torch in c:\\users\\iment\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (2.6.0)\n",
      "Requirement already satisfied: filelock in c:\\users\\iment\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from torch) (3.14.0)\n",
      "Requirement already satisfied: typing-extensions>=4.10.0 in c:\\users\\iment\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from torch) (4.11.0)\n",
      "Requirement already satisfied: networkx in c:\\users\\iment\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from torch) (3.3)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\iment\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from torch) (3.1.2)\n",
      "Requirement already satisfied: fsspec in c:\\users\\iment\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from torch) (2024.3.1)\n",
      "Requirement already satisfied: sympy==1.13.1 in c:\\users\\iment\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from torch) (1.13.1)\n",
      "Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\\users\\iment\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from sympy==1.13.1->torch) (1.3.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\iment\\appdata\\local\\programs\\python\\python310\\lib\\site-packages (from jinja2->torch) (2.0.1)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n",
      "[notice] A new release of pip is available: 24.3.1 -> 25.0.1\n",
      "[notice] To update, run: python.exe -m pip install --upgrade pip\n"
     ]
    }
   ],
   "source": [
    "##pip install torch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f45203a6-8d3d-4ed1-a66e-56e7f308ff83",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from torch.utils.data import DataLoader, TensorDataset\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "import torch.nn.functional as F\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d44fb534-137a-4708-ab2c-11c5c25f5d0e",
   "metadata": {},
   "source": [
    "### DATASET"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "412d3b4f-f1d2-4f0d-93a3-d1a9b3565910",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Unnamed: 0  Zaman  Maxell     GP  Fujitsu  Duracell\n",
      "0         NaN    0.0   3.307  3.301    3.287     3.238\n",
      "1         NaN    0.0   3.307  3.301    3.287     3.238\n",
      "2         NaN    1.0   3.307  3.301    3.287     3.240\n",
      "3         NaN    3.0   3.112  3.035    3.091     3.071\n",
      "4         NaN    5.0   3.066  3.015    3.051     3.039\n"
     ]
    }
   ],
   "source": [
    "# Load the Excel file\n",
    "file_path = 'pil_test.xlsx'  # Update the path if the file is located elsewhere\n",
    "df = pd.read_excel(file_path)\n",
    "\n",
    "# Check the first few rows of the data to ensure it's loaded correctly\n",
    "print(df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3ef6126d-bda5-449b-beb3-188fe9fdd48b",
   "metadata": {},
   "source": [
    "### PREPROCESSING"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2ce9f306-fc82-4adf-a2f5-1a76a59aae78",
   "metadata": {},
   "source": [
    "fill null values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3ccb47f2-2f5c-46db-a8a2-ac2ca1865e67",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Unnamed: 0    6370\n",
      "Zaman          315\n",
      "Maxell         317\n",
      "GP             315\n",
      "Fujitsu          0\n",
      "Duracell       372\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(df.isna().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e366e4de-ec8e-49b7-801e-f312da11a0a7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Unnamed: 0    6370\n",
      "Zaman            0\n",
      "Maxell           0\n",
      "GP               0\n",
      "Fujitsu          0\n",
      "Duracell         0\n",
      "dtype: int64\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\iment\\AppData\\Local\\Temp\\ipykernel_57500\\2946132797.py:4: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.\n",
      "  df = df.fillna(method='ffill')  # Forward fill to handle missing data\n"
     ]
    }
   ],
   "source": [
    "# df = df.drop(columns=['Unnamed: 0'])\n",
    "\n",
    "# Handle missing values (if any)\n",
    "df = df.fillna(method='ffill')  # Forward fill to handle missing data\n",
    "\n",
    "# Optionally, check for NaN values after filling\n",
    "print(df.isna().sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "10f2e27e-1bfe-4f4d-9d50-d7b1b06a388c",
   "metadata": {},
   "source": [
    "Min-Max Normalization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f548fd18-80c5-40a2-9842-3e6def55386f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1.         1.         1.         0.99861207]\n",
      " [1.         1.         1.         0.99861207]\n",
      " [1.         1.         1.         1.        ]\n",
      " [0.87060385 0.8229028  0.86819099 0.88272033]\n",
      " [0.84007963 0.80958722 0.84129119 0.86051353]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.preprocessing import MinMaxScaler\n",
    "\n",
    "# Initialize the MinMaxScaler to scale between 0 and 1\n",
    "voltage_data = df[['Maxell', 'GP', 'Fujitsu', 'Duracell']].values\n",
    "scaler = MinMaxScaler()\n",
    "\n",
    "# Apply the scaler to your data\n",
    "data_scaled = scaler.fit_transform(voltage_data)\n",
    "\n",
    "# Print the scaled data to verify it’s between 0 and 1\n",
    "print(data_scaled[:5])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "61fb0451-be2a-4885-b424-546455b21bc7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to create sequences for each battery independently\n",
    "def create_sequences(data, seq_length=67):\n",
    "    sequences = []\n",
    "    labels = []\n",
    "    for i in range(len(data) - seq_length):\n",
    "        seq = data[i:i + seq_length]\n",
    "        label = data[i + seq_length]  # Next time step value for this battery\n",
    "        sequences.append(seq)\n",
    "        labels.append(label)\n",
    "    return np.array(sequences), np.array(labels)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "9d2bef36-2eeb-4e69-8aaa-772b09dc9bbd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to extract data for a specific battery type\n",
    "def get_battery_data(battery_type, data_scaled, battery_columns, seq_length=67):\n",
    "    # Ensure the battery column exists in the DataFrame\n",
    "    if battery_type not in battery_columns:\n",
    "        raise ValueError(f\"{battery_type} not found in battery columns\")\n",
    "    \n",
    "    # Get the index of the battery column you are interested in\n",
    "    target_idx = battery_columns.index(battery_type)\n",
    "    \n",
    "    # Select the voltage data for just that battery (i.e., single column of data)\n",
    "    y_data = data_scaled[:, target_idx]  # Select target column for y\n",
    "    \n",
    "    # Create sequences for that battery's voltage data only\n",
    "    X_battery, y_battery = create_sequences(y_data, seq_length=seq_length)\n",
    "    \n",
    "    return X_battery, y_battery\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "667749d9-9cc9-44ad-8f27-5d03aa13cda0",
   "metadata": {},
   "source": [
    "here we can choose which battery to use by giving battery type"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "df5b5324-9811-4793-a4a4-cc1ea2a87bee",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Battery columns\n",
    "battery_columns = ['Maxell', 'GP', 'Fujitsu', 'Duracell']\n",
    "\n",
    "# Example usage: Extract data for 'Maxell'\n",
    "battery_type = 'Duracell'  # Specify the battery type you want data for\n",
    "\n",
    "X_battery, y_battery = get_battery_data(battery_type, data_scaled, battery_columns)\n",
    "\n",
    "# Print the shapes of the extracted data\n",
    "# print(f\"X_battery shape for {battery_type}: {X_battery}\")\n",
    "# print(f\"y_battery shape for {battery_type}: {y_battery}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "950e90fa-d497-4dc5-94a0-54939bc57b4b",
   "metadata": {},
   "source": [
    "splitting data to train val and test dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "de156600-6bf3-427e-95fb-a34ac36fd3af",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(5042, 67) (630, 67) (631, 67) (5042,) (630,) (631,)\n"
     ]
    }
   ],
   "source": [
    "# Split sizes\n",
    "train_size = int(len(X_battery) * 0.8)  # 80% for training\n",
    "val_size = int(len(X_battery) * 0.1)    # 10% for validation\n",
    "test_size = len(X_battery) - train_size - val_size  # The rest will be for testing\n",
    "\n",
    "# Split into training, validation, and test sets\n",
    "X_train, X_val, X_test = X_battery[:train_size], X_battery[train_size:train_size + val_size], X_battery[train_size + val_size:]\n",
    "y_train, y_val, y_test = y_battery[:train_size], y_battery[train_size:train_size + val_size], y_battery[train_size + val_size:]\n",
    "\n",
    "# Check the split sizes\n",
    "print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "fd15c826-b211-4ef6-8935-52871d436273",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_tensor = torch.tensor(X_train, dtype=torch.float32)\n",
    "X_val_tensor = torch.tensor(X_val, dtype=torch.float32)\n",
    "X_test_tensor = torch.tensor(X_test, dtype=torch.float32)\n",
    "y_train_tensor = torch.tensor(y_train, dtype=torch.float32)\n",
    "y_val_tensor = torch.tensor(y_val, dtype=torch.float32)\n",
    "y_test_tensor = torch.tensor(y_test, dtype=torch.float32)\n",
    "\n",
    "# Move tensors to GPU if available\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "X_train_tensor = X_train_tensor.to(device)\n",
    "X_val_tensor = X_val_tensor.to(device)\n",
    "X_test_tensor = X_test_tensor.to(device)\n",
    "y_train_tensor = y_train_tensor.to(device)\n",
    "y_val_tensor = y_val_tensor.to(device)\n",
    "y_test_tensor = y_test_tensor.to(device)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8d9fb8be-8a7b-4612-aeb3-06b74682d5fc",
   "metadata": {},
   "source": [
    "### MODEL"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "915ad2ac-6ae9-441f-8b6c-76524a2238a0",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SOHMode(\n",
       "  (pre_net): Linear(in_features=67, out_features=512, bias=True)\n",
       "  (backbone): LSTM(\n",
       "    (net): LSTM(4, 128, num_layers=2, batch_first=True)\n",
       "    (predictor): Sequential(\n",
       "      (0): Linear(in_features=128, out_features=64, bias=True)\n",
       "      (1): LeakyReLU(negative_slope=0.01)\n",
       "      (2): Linear(in_features=64, out_features=1, bias=True)\n",
       "    )\n",
       "  )\n",
       "  (mse): MSELoss()\n",
       ")"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "PATH = 'LSTM_model.pth'  # Path to the saved weights\n",
    "model = torch.load(PATH, map_location=torch.device('cpu'),weights_only=False)\n",
    "\n",
    "# Set the model to evaluation mode\n",
    "model.eval()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "defcb44c-39ed-4554-bae6-99aa8721242f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "SOHMode(\n",
       "  (pre_net): Linear(in_features=67, out_features=512, bias=True)\n",
       "  (backbone): LSTM(\n",
       "    (net): LSTM(4, 128, num_layers=2, batch_first=True)\n",
       "    (predictor): Sequential(\n",
       "      (0): Linear(in_features=128, out_features=64, bias=True)\n",
       "      (1): LeakyReLU(negative_slope=0.01)\n",
       "      (2): Linear(in_features=64, out_features=1, bias=True)\n",
       "    )\n",
       "  )\n",
       "  (mse): MSELoss()\n",
       ")"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model.pre_net = nn.Linear(67, 512)\n",
    "\n",
    "# Set the model to eval mode\n",
    "model.eval()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "4c868890-892b-484b-aa37-66e6bf9f6c37",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "# Define the fine-tuned model with LSTM\n",
    "# class FineTunedModel(nn.Module):\n",
    "#     def __init__(self, pretrained_lstm):\n",
    "#         super(FineTunedModel, self).__init__()\n",
    "#         self.pretrained_lstm = pretrained_lstm\n",
    "#         self.fc = nn.Linear(128, 1)  # Output layer to predict one value\n",
    "\n",
    "  \n",
    "#     def forward(self, x):\n",
    "#         # Unpack the output of the LSTM\n",
    "#         lstm_out, (h_n, c_n) = self.pretrained_lstm(x)\n",
    "\n",
    "#         # Get the output of the last time step\n",
    "#         if lstm_out.dim() == 3:  # In case the sequence length is greater than 1\n",
    "#             last_hidden_state = lstm_out[:, -1, :]  # Last time step output\n",
    "#         else:  # If sequence length is 1\n",
    "#             last_hidden_state = lstm_out\n",
    "\n",
    "#         # Pass the last hidden state through the fully connected layer\n",
    "#         output = self.fc(last_hidden_state)\n",
    "        \n",
    "#         return output\n",
    "\n",
    "class FineTunedModel(nn.Module):\n",
    "    def __init__(self, pretrained_lstm, freeze_lstm=True):\n",
    "        super(FineTunedModel, self).__init__()\n",
    "        self.pre_net = nn.Linear(67, 512)         # Match original model's pre_net\n",
    "        self.project = nn.Linear(512, 4)          # Match expected input of LSTM\n",
    "        self.pretrained_lstm = pretrained_lstm    # This is model.backbone.net (LSTM)\n",
    "        self.fc = nn.Linear(128, 1)               # Final output layer\n",
    "\n",
    "        if freeze_lstm:\n",
    "            for param in self.pretrained_lstm.parameters():\n",
    "                param.requires_grad = False\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.pre_net(x)       # 67 → 512\n",
    "        x = self.project(x)       # 512 → 4\n",
    "        lstm_out, (h_n, c_n) = self.pretrained_lstm(x)  # 4 → 128\n",
    "\n",
    "        # Use last time step output\n",
    "        last_hidden = lstm_out[:, -1, :] if lstm_out.dim() == 3 else lstm_out\n",
    "        return self.fc(last_hidden)\n",
    "\n",
    "\n",
    "# Assuming the pre-trained LSTM has been defined as follows:\n",
    "# pretrained_lstm_model = nn.LSTM(input_size=67, hidden_size=128, num_layers=2, batch_first=True)\n",
    "pretrained_lstm = model.backbone.net\n",
    "# Initialize the fine-tuned model\n",
    "fine_tuned_model = FineTunedModel(pretrained_lstm)\n",
    "\n",
    "# Freeze the LSTM layers if you want to fine-tune only the fully connected layer\n",
    "# for param in fine_tuned_model.pretrained_lstm.parameters():\n",
    "#     param.requires_grad = False  # Freeze LSTM layers\n",
    "\n",
    "# Define the loss function and optimizer\n",
    "criterion = nn.MSELoss()\n",
    "optimizer = optim.Adam(fine_tuned_model.parameters(), lr=0.001)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "1e2232fb-6bb1-483a-9534-b3a2d1599109",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "FineTunedModel(\n",
       "  (pre_net): Linear(in_features=67, out_features=512, bias=True)\n",
       "  (project): Linear(in_features=512, out_features=4, bias=True)\n",
       "  (pretrained_lstm): LSTM(4, 128, num_layers=2, batch_first=True)\n",
       "  (fc): Linear(in_features=128, out_features=1, bias=True)\n",
       ")"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "fine_tuned_model.eval()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "9cae7ce0-4fa1-41e0-8ac9-4869c9c996f5",
   "metadata": {},
   "outputs": [],
   "source": [
    "##pip install openpyxl==3.1.0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "c9c0bc27-220a-4203-9ef1-c3aaf88c08ce",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# print(fine_tuned_model.backbone)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "e45a35a2-6988-465f-b894-e267658a0ae2",
   "metadata": {},
   "outputs": [],
   "source": [
    "# last_layer = model.backbone.predictor\n",
    "\n",
    "# print(last_layer)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "5c7a0aa2-41a9-4c66-a063-02c3f43815c7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([4963, 67]) torch.Size([4963])\n"
     ]
    }
   ],
   "source": [
    "print(X_train_tensor.shape,y_train_tensor.shape)\n",
    "dataset = TensorDataset(X_train_tensor, y_train_tensor)\n",
    "loader = DataLoader(dataset, batch_size=64, shuffle=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "88679e6e-b4de-4d6b-9c10-22ef091ae9c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)\n",
    "val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor), batch_size=64, shuffle=False)\n",
    "test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "1959c655-9ad0-46d0-acd5-cc4799bf0c80",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [1/50], Train Loss: 0.0381, Train MAE: 0.1527, Train MSE: 0.0381, Train RMSE: 0.1900, Train R²: -0.6350\n",
      "Validation - Loss: 0.0363, MAE: 0.1477, MSE: 0.0363, RMSE: 0.1837, R²: -0.4055\n",
      "Epoch [2/50], Train Loss: 0.0285, Train MAE: 0.1324, Train MSE: 0.0285, Train RMSE: 0.1683, Train R²: -0.2107\n",
      "Validation - Loss: 0.0348, MAE: 0.1452, MSE: 0.0348, RMSE: 0.1803, R²: -0.3495\n",
      "Epoch [3/50], Train Loss: 0.0279, Train MAE: 0.1319, Train MSE: 0.0279, Train RMSE: 0.1667, Train R²: -0.1908\n",
      "Validation - Loss: 0.0336, MAE: 0.1429, MSE: 0.0336, RMSE: 0.1777, R²: -0.3068\n",
      "Epoch [4/50], Train Loss: 0.0279, Train MAE: 0.1320, Train MSE: 0.0279, Train RMSE: 0.1666, Train R²: -0.1731\n",
      "Validation - Loss: 0.0325, MAE: 0.1386, MSE: 0.0325, RMSE: 0.1753, R²: -0.2673\n",
      "Epoch [5/50], Train Loss: 0.0267, Train MAE: 0.1297, Train MSE: 0.0267, Train RMSE: 0.1628, Train R²: -0.1298\n",
      "Validation - Loss: 0.0322, MAE: 0.1420, MSE: 0.0322, RMSE: 0.1744, R²: -0.2544\n",
      "Epoch [6/50], Train Loss: 0.0267, Train MAE: 0.1301, Train MSE: 0.0267, Train RMSE: 0.1630, Train R²: -0.1192\n",
      "Validation - Loss: 0.0314, MAE: 0.1408, MSE: 0.0314, RMSE: 0.1727, R²: -0.2272\n",
      "Epoch [7/50], Train Loss: 0.0264, Train MAE: 0.1293, Train MSE: 0.0264, Train RMSE: 0.1619, Train R²: -0.1142\n",
      "Validation - Loss: 0.0314, MAE: 0.1439, MSE: 0.0314, RMSE: 0.1730, R²: -0.2313\n",
      "Epoch [8/50], Train Loss: 0.0262, Train MAE: 0.1286, Train MSE: 0.0262, Train RMSE: 0.1614, Train R²: -0.1054\n",
      "Validation - Loss: 0.0302, MAE: 0.1397, MSE: 0.0302, RMSE: 0.1699, R²: -0.1844\n",
      "Epoch [9/50], Train Loss: 0.0257, Train MAE: 0.1277, Train MSE: 0.0257, Train RMSE: 0.1598, Train R²: -0.0840\n",
      "Validation - Loss: 0.0291, MAE: 0.1333, MSE: 0.0291, RMSE: 0.1671, R²: -0.1418\n",
      "Epoch [10/50], Train Loss: 0.0254, Train MAE: 0.1278, Train MSE: 0.0254, Train RMSE: 0.1589, Train R²: -0.0677\n",
      "Validation - Loss: 0.0285, MAE: 0.1315, MSE: 0.0285, RMSE: 0.1657, R²: -0.1202\n",
      "Epoch [11/50], Train Loss: 0.0249, Train MAE: 0.1257, Train MSE: 0.0249, Train RMSE: 0.1573, Train R²: -0.0472\n",
      "Validation - Loss: 0.0279, MAE: 0.1331, MSE: 0.0279, RMSE: 0.1640, R²: -0.0971\n",
      "Epoch [12/50], Train Loss: 0.0244, Train MAE: 0.1251, Train MSE: 0.0244, Train RMSE: 0.1558, Train R²: -0.0314\n",
      "Validation - Loss: 0.0272, MAE: 0.1311, MSE: 0.0272, RMSE: 0.1620, R²: -0.0694\n",
      "Epoch [13/50], Train Loss: 0.0241, Train MAE: 0.1236, Train MSE: 0.0241, Train RMSE: 0.1547, Train R²: -0.0072\n",
      "Validation - Loss: 0.0266, MAE: 0.1296, MSE: 0.0266, RMSE: 0.1605, R²: -0.0482\n",
      "Epoch [14/50], Train Loss: 0.0238, Train MAE: 0.1227, Train MSE: 0.0238, Train RMSE: 0.1537, Train R²: 0.0039\n",
      "Validation - Loss: 0.0263, MAE: 0.1299, MSE: 0.0263, RMSE: 0.1597, R²: -0.0372\n",
      "Epoch [15/50], Train Loss: 0.0234, Train MAE: 0.1220, Train MSE: 0.0234, Train RMSE: 0.1526, Train R²: 0.0180\n",
      "Validation - Loss: 0.0258, MAE: 0.1257, MSE: 0.0258, RMSE: 0.1583, R²: -0.0184\n",
      "Epoch [16/50], Train Loss: 0.0236, Train MAE: 0.1218, Train MSE: 0.0236, Train RMSE: 0.1530, Train R²: 0.0124\n",
      "Validation - Loss: 0.0255, MAE: 0.1248, MSE: 0.0255, RMSE: 0.1577, R²: -0.0096\n",
      "Epoch [17/50], Train Loss: 0.0233, Train MAE: 0.1213, Train MSE: 0.0233, Train RMSE: 0.1519, Train R²: 0.0282\n",
      "Validation - Loss: 0.0252, MAE: 0.1260, MSE: 0.0252, RMSE: 0.1566, R²: 0.0048\n",
      "Epoch [18/50], Train Loss: 0.0229, Train MAE: 0.1199, Train MSE: 0.0229, Train RMSE: 0.1506, Train R²: 0.0361\n",
      "Validation - Loss: 0.0250, MAE: 0.1261, MSE: 0.0250, RMSE: 0.1562, R²: 0.0104\n",
      "Epoch [19/50], Train Loss: 0.0232, Train MAE: 0.1213, Train MSE: 0.0232, Train RMSE: 0.1516, Train R²: 0.0274\n",
      "Validation - Loss: 0.0247, MAE: 0.1237, MSE: 0.0247, RMSE: 0.1553, R²: 0.0218\n",
      "Epoch [20/50], Train Loss: 0.0230, Train MAE: 0.1205, Train MSE: 0.0230, Train RMSE: 0.1510, Train R²: 0.0373\n",
      "Validation - Loss: 0.0246, MAE: 0.1236, MSE: 0.0246, RMSE: 0.1550, R²: 0.0260\n",
      "Epoch [21/50], Train Loss: 0.0226, Train MAE: 0.1194, Train MSE: 0.0226, Train RMSE: 0.1498, Train R²: 0.0460\n",
      "Validation - Loss: 0.0244, MAE: 0.1224, MSE: 0.0244, RMSE: 0.1546, R²: 0.0311\n",
      "Epoch [22/50], Train Loss: 0.0227, Train MAE: 0.1192, Train MSE: 0.0227, Train RMSE: 0.1502, Train R²: 0.0458\n",
      "Validation - Loss: 0.0243, MAE: 0.1237, MSE: 0.0243, RMSE: 0.1544, R²: 0.0341\n",
      "Epoch [23/50], Train Loss: 0.0224, Train MAE: 0.1187, Train MSE: 0.0224, Train RMSE: 0.1492, Train R²: 0.0517\n",
      "Validation - Loss: 0.0243, MAE: 0.1229, MSE: 0.0243, RMSE: 0.1542, R²: 0.0369\n",
      "Epoch [24/50], Train Loss: 0.0228, Train MAE: 0.1194, Train MSE: 0.0228, Train RMSE: 0.1503, Train R²: 0.0453\n",
      "Validation - Loss: 0.0242, MAE: 0.1230, MSE: 0.0242, RMSE: 0.1540, R²: 0.0385\n",
      "Epoch [25/50], Train Loss: 0.0229, Train MAE: 0.1198, Train MSE: 0.0229, Train RMSE: 0.1509, Train R²: 0.0417\n",
      "Validation - Loss: 0.0248, MAE: 0.1288, MSE: 0.0248, RMSE: 0.1561, R²: 0.0120\n",
      "Epoch [26/50], Train Loss: 0.0229, Train MAE: 0.1204, Train MSE: 0.0229, Train RMSE: 0.1507, Train R²: 0.0400\n",
      "Validation - Loss: 0.0241, MAE: 0.1219, MSE: 0.0241, RMSE: 0.1537, R²: 0.0431\n",
      "Epoch [27/50], Train Loss: 0.0229, Train MAE: 0.1196, Train MSE: 0.0229, Train RMSE: 0.1506, Train R²: 0.0438\n",
      "Validation - Loss: 0.0242, MAE: 0.1200, MSE: 0.0242, RMSE: 0.1542, R²: 0.0364\n",
      "Epoch [28/50], Train Loss: 0.0228, Train MAE: 0.1193, Train MSE: 0.0228, Train RMSE: 0.1504, Train R²: 0.0432\n",
      "Validation - Loss: 0.0241, MAE: 0.1232, MSE: 0.0241, RMSE: 0.1538, R²: 0.0416\n",
      "Epoch [29/50], Train Loss: 0.0227, Train MAE: 0.1190, Train MSE: 0.0227, Train RMSE: 0.1501, Train R²: 0.0463\n",
      "Validation - Loss: 0.0241, MAE: 0.1213, MSE: 0.0241, RMSE: 0.1536, R²: 0.0441\n",
      "Epoch [30/50], Train Loss: 0.0228, Train MAE: 0.1199, Train MSE: 0.0228, Train RMSE: 0.1506, Train R²: 0.0477\n",
      "Validation - Loss: 0.0241, MAE: 0.1202, MSE: 0.0241, RMSE: 0.1536, R²: 0.0441\n",
      "Epoch [31/50], Train Loss: 0.0229, Train MAE: 0.1190, Train MSE: 0.0229, Train RMSE: 0.1506, Train R²: 0.0418\n",
      "Validation - Loss: 0.0242, MAE: 0.1244, MSE: 0.0242, RMSE: 0.1540, R²: 0.0391\n",
      "Epoch [32/50], Train Loss: 0.0229, Train MAE: 0.1194, Train MSE: 0.0229, Train RMSE: 0.1508, Train R²: 0.0389\n",
      "Validation - Loss: 0.0240, MAE: 0.1226, MSE: 0.0240, RMSE: 0.1534, R²: 0.0465\n",
      "Epoch [33/50], Train Loss: 0.0227, Train MAE: 0.1189, Train MSE: 0.0227, Train RMSE: 0.1501, Train R²: 0.0458\n",
      "Validation - Loss: 0.0240, MAE: 0.1215, MSE: 0.0240, RMSE: 0.1534, R²: 0.0473\n",
      "Epoch [34/50], Train Loss: 0.0227, Train MAE: 0.1194, Train MSE: 0.0227, Train RMSE: 0.1497, Train R²: 0.0519\n",
      "Validation - Loss: 0.0241, MAE: 0.1196, MSE: 0.0241, RMSE: 0.1538, R²: 0.0417\n",
      "Epoch [35/50], Train Loss: 0.0227, Train MAE: 0.1194, Train MSE: 0.0227, Train RMSE: 0.1501, Train R²: 0.0456\n",
      "Validation - Loss: 0.0239, MAE: 0.1200, MSE: 0.0239, RMSE: 0.1532, R²: 0.0492\n",
      "Epoch [36/50], Train Loss: 0.0226, Train MAE: 0.1185, Train MSE: 0.0226, Train RMSE: 0.1497, Train R²: 0.0498\n",
      "Validation - Loss: 0.0241, MAE: 0.1236, MSE: 0.0241, RMSE: 0.1538, R²: 0.0411\n",
      "Epoch [37/50], Train Loss: 0.0225, Train MAE: 0.1186, Train MSE: 0.0225, Train RMSE: 0.1494, Train R²: 0.0506\n",
      "Validation - Loss: 0.0241, MAE: 0.1197, MSE: 0.0241, RMSE: 0.1537, R²: 0.0433\n",
      "Epoch [38/50], Train Loss: 0.0228, Train MAE: 0.1192, Train MSE: 0.0228, Train RMSE: 0.1502, Train R²: 0.0415\n",
      "Validation - Loss: 0.0240, MAE: 0.1236, MSE: 0.0240, RMSE: 0.1536, R²: 0.0443\n",
      "Epoch [39/50], Train Loss: 0.0225, Train MAE: 0.1184, Train MSE: 0.0225, Train RMSE: 0.1497, Train R²: 0.0499\n",
      "Validation - Loss: 0.0246, MAE: 0.1280, MSE: 0.0246, RMSE: 0.1554, R²: 0.0196\n",
      "Epoch [40/50], Train Loss: 0.0226, Train MAE: 0.1185, Train MSE: 0.0226, Train RMSE: 0.1497, Train R²: 0.0499\n",
      "Validation - Loss: 0.0242, MAE: 0.1184, MSE: 0.0242, RMSE: 0.1541, R²: 0.0374\n",
      "Epoch [41/50], Train Loss: 0.0226, Train MAE: 0.1184, Train MSE: 0.0226, Train RMSE: 0.1498, Train R²: 0.0522\n",
      "Validation - Loss: 0.0240, MAE: 0.1235, MSE: 0.0240, RMSE: 0.1535, R²: 0.0452\n",
      "Epoch [42/50], Train Loss: 0.0228, Train MAE: 0.1192, Train MSE: 0.0228, Train RMSE: 0.1505, Train R²: 0.0454\n",
      "Validation - Loss: 0.0242, MAE: 0.1179, MSE: 0.0242, RMSE: 0.1541, R²: 0.0385\n",
      "Epoch [43/50], Train Loss: 0.0227, Train MAE: 0.1187, Train MSE: 0.0227, Train RMSE: 0.1499, Train R²: 0.0510\n",
      "Validation - Loss: 0.0239, MAE: 0.1211, MSE: 0.0239, RMSE: 0.1532, R²: 0.0493\n",
      "Epoch [44/50], Train Loss: 0.0226, Train MAE: 0.1187, Train MSE: 0.0226, Train RMSE: 0.1499, Train R²: 0.0514\n",
      "Validation - Loss: 0.0239, MAE: 0.1220, MSE: 0.0239, RMSE: 0.1531, R²: 0.0506\n",
      "Epoch [45/50], Train Loss: 0.0227, Train MAE: 0.1184, Train MSE: 0.0227, Train RMSE: 0.1499, Train R²: 0.0417\n",
      "Validation - Loss: 0.0240, MAE: 0.1227, MSE: 0.0240, RMSE: 0.1533, R²: 0.0472\n",
      "Epoch [46/50], Train Loss: 0.0226, Train MAE: 0.1186, Train MSE: 0.0226, Train RMSE: 0.1497, Train R²: 0.0494\n",
      "Validation - Loss: 0.0238, MAE: 0.1202, MSE: 0.0238, RMSE: 0.1528, R²: 0.0541\n",
      "Epoch [47/50], Train Loss: 0.0225, Train MAE: 0.1179, Train MSE: 0.0225, Train RMSE: 0.1494, Train R²: 0.0541\n",
      "Validation - Loss: 0.0240, MAE: 0.1240, MSE: 0.0240, RMSE: 0.1535, R²: 0.0449\n",
      "Epoch [48/50], Train Loss: 0.0227, Train MAE: 0.1188, Train MSE: 0.0227, Train RMSE: 0.1502, Train R²: 0.0512\n",
      "Validation - Loss: 0.0239, MAE: 0.1227, MSE: 0.0239, RMSE: 0.1531, R²: 0.0503\n",
      "Epoch [49/50], Train Loss: 0.0225, Train MAE: 0.1183, Train MSE: 0.0225, Train RMSE: 0.1496, Train R²: 0.0591\n",
      "Validation - Loss: 0.0239, MAE: 0.1188, MSE: 0.0239, RMSE: 0.1531, R²: 0.0503\n",
      "Epoch [50/50], Train Loss: 0.0226, Train MAE: 0.1186, Train MSE: 0.0226, Train RMSE: 0.1499, Train R²: 0.0475\n",
      "Validation - Loss: 0.0239, MAE: 0.1219, MSE: 0.0239, RMSE: 0.1531, R²: 0.0508\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Assuming your fine_tuned_model, criterion (e.g., MSELoss), and optimizer are already defined\n",
    "num_epochs = 50\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "fine_tuned_model.to(device)\n",
    "\n",
    "# Function to calculate R-squared (R²)\n",
    "def r2_score(y_true, y_pred):\n",
    "    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)\n",
    "    ss_residual = torch.sum((y_true - y_pred) ** 2)\n",
    "    return 1 - (ss_residual / ss_total)\n",
    "\n",
    "# Function to evaluate the model\n",
    "def evaluate_model(model, loader, device):\n",
    "    model.eval()  # Set model to evaluation mode\n",
    "    epoch_loss = 0.0\n",
    "    epoch_mae = 0.0\n",
    "    epoch_mse = 0.0\n",
    "    epoch_rmse = 0.0\n",
    "    epoch_r2 = 0.0\n",
    "\n",
    "    with torch.no_grad():  # Disable gradient computation for evaluation\n",
    "        for X_batch, y_batch in loader:\n",
    "            X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n",
    "            outputs = model(X_batch)\n",
    "\n",
    "            # Ensure output and target shapes match\n",
    "            if outputs.shape != y_batch.shape:\n",
    "                y_batch = y_batch.view_as(outputs)\n",
    "\n",
    "            # Calculate metrics\n",
    "            loss = criterion(outputs, y_batch)\n",
    "            mae = torch.mean(torch.abs(outputs - y_batch))  # Mean Absolute Error\n",
    "            mse = F.mse_loss(outputs, y_batch)  # Mean Squared Error\n",
    "            rmse = torch.sqrt(mse)  # Root Mean Squared Error\n",
    "            r2 = r2_score(y_batch, outputs)  # R-squared\n",
    "\n",
    "            # Accumulate metrics for the epoch\n",
    "            epoch_loss += loss.item()\n",
    "            epoch_mae += mae.item()\n",
    "            epoch_mse += mse.item()\n",
    "            epoch_rmse += rmse.item()\n",
    "            epoch_r2 += r2.item()\n",
    "\n",
    "    # Return average metrics for the epoch\n",
    "    return epoch_loss / len(loader), epoch_mae / len(loader), epoch_mse / len(loader), epoch_rmse / len(loader), epoch_r2 / len(loader)\n",
    "\n",
    "# Training loop\n",
    "for epoch in range(num_epochs):\n",
    "    fine_tuned_model.train()  # Set model to training mode\n",
    "    epoch_loss = 0.0\n",
    "    epoch_mae = 0.0\n",
    "    epoch_mse = 0.0\n",
    "    epoch_rmse = 0.0\n",
    "    epoch_r2 = 0.0\n",
    "    \n",
    "    # Training phase\n",
    "    for X_batch, y_batch in loader:\n",
    "        X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n",
    "\n",
    "        optimizer.zero_grad()  # Zero the gradients\n",
    "        outputs = fine_tuned_model(X_batch)  # Forward pass\n",
    "\n",
    "        # Ensure output and target shapes match\n",
    "        if outputs.shape != y_batch.shape:\n",
    "            y_batch = y_batch.view_as(outputs)\n",
    "\n",
    "        # Calculate loss\n",
    "        loss = criterion(outputs, y_batch)  # Calculate MSE loss (since criterion is usually MSELoss)\n",
    "\n",
    "        # Calculate MAE and MSE\n",
    "        mae = torch.mean(torch.abs(outputs - y_batch))  # Mean Absolute Error\n",
    "        mse = F.mse_loss(outputs, y_batch)  # Mean Squared Error\n",
    "        rmse = torch.sqrt(mse)  # Root Mean Squared Error (RMSE)\n",
    "        r2 = r2_score(y_batch, outputs)  # R-squared (R²)\n",
    "\n",
    "        loss.backward()  # Backward pass\n",
    "        optimizer.step()  # Update weights\n",
    "\n",
    "        # Accumulate loss and metrics for the epoch\n",
    "        epoch_loss += loss.item()\n",
    "        epoch_mae += mae.item()\n",
    "        epoch_mse += mse.item()\n",
    "        epoch_rmse += rmse.item()\n",
    "        epoch_r2 += r2.item()\n",
    "\n",
    "    # Print the average training metrics for the epoch\n",
    "    print(f\"Epoch [{epoch+1}/{num_epochs}], \"\n",
    "          f\"Train Loss: {epoch_loss/len(loader):.4f}, \"\n",
    "          f\"Train MAE: {epoch_mae/len(loader):.4f}, \"\n",
    "          f\"Train MSE: {epoch_mse/len(loader):.4f}, \"\n",
    "          f\"Train RMSE: {epoch_rmse/len(loader):.4f}, \"\n",
    "          f\"Train R²: {epoch_r2/len(loader):.4f}\")\n",
    "\n",
    "    # Validation phase after each epoch\n",
    "    val_loss, val_mae, val_mse, val_rmse, val_r2 = evaluate_model(fine_tuned_model, val_loader, device)\n",
    "    print(f\"Validation - Loss: {val_loss:.4f}, \"\n",
    "          f\"MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, \"\n",
    "          f\"RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "3a4d67cd-a1e7-4abe-b0bf-ef7ed654e0b7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation - MSE: 0.0254, MAE: 0.1241, RMSE: 0.1593, R²: -733.5850\n",
      "Test - MSE: 0.0256, MAE: 0.1265, RMSE: 0.1602, R²: -651.9230\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\iment\\AppData\\Local\\Temp\\ipykernel_57500\\2746724341.py:13: UserWarning: Using a target size (torch.Size([709])) that is different to the input size (torch.Size([709, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.\n",
      "  mse = F.mse_loss(outputs, y_data)\n",
      "C:\\Users\\iment\\AppData\\Local\\Temp\\ipykernel_57500\\2746724341.py:13: UserWarning: Using a target size (torch.Size([631])) that is different to the input size (torch.Size([631, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.\n",
      "  mse = F.mse_loss(outputs, y_data)\n"
     ]
    }
   ],
   "source": [
    "fine_tuned_model.eval()  # Set the model to evaluation mode\n",
    "\n",
    "# # Prepare validation and test sets as tensors\n",
    "# X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)\n",
    "# y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)\n",
    "# X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)\n",
    "# y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)\n",
    "\n",
    "# Perform validation and test evaluations\n",
    "def evaluate_model(model, X_data, y_data):\n",
    "    with torch.no_grad():\n",
    "        outputs = model(X_data)\n",
    "        mse = F.mse_loss(outputs, y_data)\n",
    "        mae = torch.mean(torch.abs(outputs - y_data))\n",
    "        rmse = torch.sqrt(mse)\n",
    "        r2 = r2_score(y_data, outputs)\n",
    "        return mse.item(), mae.item(), rmse.item(), r2.item()\n",
    "\n",
    "# Evaluate on the validation set\n",
    "val_mse, val_mae, val_rmse, val_r2 = evaluate_model(fine_tuned_model, X_val_tensor, y_val_tensor)\n",
    "print(f\"Validation - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}\")\n",
    "\n",
    "# Evaluate on the test set\n",
    "test_mse, test_mae, test_rmse, test_r2 = evaluate_model(fine_tuned_model, X_test_tensor, y_test_tensor)\n",
    "print(f\"Test - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "3b41b852-36ff-4fd2-ada0-cbea4207cb64",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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lYulpaXJ3tSo6ijTre7Hjh2LHTt24OLFi9Dr9ahduzbWr1+Pfv36AQDOnz8Pf39/JCUloXXr1ti9ezd69OiBmzdvSkdVli9fjkmTJuH27duwsbEp0X55q3sikhve6p7o6SrkVve5ubn46quv8Oabb0KhUOD48eN48OABgoODpT5+fn6oU6cOkpKSAABJSUlo3LixwSmfkJAQ6PV6nDlz5rH7ysnJgV6vN3gQERFR1VXqgLJ161ZkZGRgyJAhAACdTgcbGxvpY6MLubm5QafTSX0eDieFywuXPU5sbCzUarX08PT0LG3ZREREVAmUOqCsWrUK3bp1g4eHhynrKVZ0dDQyMzOlx/Xr18t9n0RERGQ+pbp05tq1a9i3bx++++47qU2j0SA3NxcZGRkGR1HS0tKky400Gg2OHj1qsK20tDRp2eMolUoolcrSlEpERESVUKmOoMTHx8PV1RWhoaFSW2BgIKytrZGYmCi1JScnIzU1FVqtFgCg1Wpx6tQpg2veExISoFKpEBAQUNoxEBERURVj9BGUgoICxMfHIyIiwuDeJWq1GkOHDsX48ePh4uIClUqFMWPGQKvVonXr1gCALl26ICAgAIMHD0ZcXBx0Oh0mT56MyMhIHiEhIiIiidEBZd++fUhNTcWbb75ZZNm8efNgYWGBsLAwgxu1FbK0tMSOHTswatQoaLVaODg4ICIiAjExMWUbBREREVUpZboPirnwPihEJDe8DwrR01XIfVCIiIiIygsDChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyY7RAeXPP//Ea6+9hpo1a8LOzg6NGzfGr7/+Ki0XQmDKlClwd3eHnZ0dgoODcfHiRYNt3L17F+Hh4VCpVHB2dsbQoUORlZVV9tEQERFRlWBUQPnf//6Htm3bwtraGrt378bZs2fx6aefokaNGlKfuLg4LFy4EMuXL8eRI0fg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oBAREZHsMKAQERGR7DCgEBERkewwoBAREZHsMKAQERGR7DCgEBERkewwoBAREZHsMKAQERGR7DCgEBERkewwoBAREZHsMKAQERGR7FiZuwA5mpdwodTrjuvcwISVEBERVU88gkJERESyw4BCREREssOAQkRERLLDgEJERESyw4BCREREssOreIioSinLVXhEJB88gkJERESyw4BCREREssOAQkRERLLDgEJERESyw4BCREREssOAQkRERLJjVEBZtmwZmjRpApVKBZVKBa1Wi927d0vLs7OzERkZiZo1a8LR0RFhYWFIS0sz2EZqaipCQ0Nhb28PV1dXTJw4EXl5eaYZDREREVUJRgWUZ599FrNmzcLx48fx66+/omPHjujduzfOnDkDABg3bhy2b9+OTZs24ccff8TNmzfRt29faf38/HyEhoYiNzcXhw4dwpo1a7B69WpMmTLFtKMiIiKiSk0hhBBl2YCLiwtmz56Nfv36oXbt2li/fj369esHADh//jz8/f2RlJSE1q1bY/fu3ejRowdu3rwJNzc3AMDy5csxadIk3L59GzY2NiXap16vh1qtRmZmJlQqVVnKL1ZZbvQ0rnMDE1ZCRMaqjDdq498Nqi6Mef8u9RyU/Px8bNiwAffv34dWq8Xx48fx4MEDBAcHS338/PxQp04dJCUlAQCSkpLQuHFjKZwAQEhICPR6vXQUpjg5OTnQ6/UGDyIiIqq6jA4op06dgqOjI5RKJf7zn/9gy5YtCAgIgE6ng42NDZydnQ36u7m5QafTAQB0Op1BOClcXrjscWJjY6FWq6WHp6ensWUTERFRJWJ0QHnuuedw8uRJHDlyBKNGjUJERATOnj1bHrVJoqOjkZmZKT2uX79ervsjIiIi8zL6wwJtbGxQv359AEBgYCCOHTuGBQsWYMCAAcjNzUVGRobBUZS0tDRoNBoAgEajwdGjRw22V3iVT2Gf4iiVSiiVSmNLJSIz4lwuIiqLMt8HpaCgADk5OQgMDIS1tTUSExOlZcnJyUhNTYVWqwUAaLVanDp1Cunp6VKfhIQEqFQqBAQElLUUIiIiqiKMOoISHR2Nbt26oU6dOrh37x7Wr1+PH374AXv37oVarcbQoUMxfvx4uLi4QKVSYcyYMdBqtWjdujUAoEuXLggICMDgwYMRFxcHnU6HyZMnIzIykkdIiIiISGJUQElP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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "plt.hist(y_train, bins=30, alpha=0.5, label='Train')\n",
    "plt.hist(y_val, bins=30, alpha=0.5, label='Validation')\n",
    "plt.hist(y_test, bins=30, alpha=0.5, label='Test')\n",
    "plt.legend()\n",
    "plt.title(\"Target Distribution Comparison\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1ea534bc-8a31-4fc6-80d3-c633560d1941",
   "metadata": {},
   "source": [
    "Dağılım Grafiği Analizi\n",
    "Grafikte çok net bir veri dengesizliği görülüyor:\n",
    "\n",
    "💙 Eğitim (Train) verisi:\n",
    "Genellikle 2.3 – 2.6 aralığında yoğunlaşıyor\n",
    "\n",
    "Geniş bir dağılıma sahip (daha fazla çeşitlilik)\n",
    "\n",
    "🟧 Doğrulama (Validation) verisi:\n",
    "1.9 – 2.05 arasında sıkışmış\n",
    "\n",
    "Dağılım dar, çeşitlilik az\n",
    "\n",
    "💚 Test verisi:\n",
    "Neredeyse tamamı 1.83 civarında sabitlenmiş\n",
    "\n",
    "Hiçbir çeşitlilik yok gibi\n",
    "\n",
    "❌ Neden R² Bu Kadar Kötü?\n",
    "R² skoru, gerçek değerlerin varyansını (dağılımını) kullanır.\n",
    "\n",
    "Eğer gerçek değerler çok az değişiyorsa (yani sabitse),\n",
    "\n",
    "Modelin hatası küçük olsa bile, R² çok kötü çıkar (hatta eksi)\n",
    "\n",
    "Yani:\n",
    "\n",
    "mathematica\n",
    "Copy code\n",
    "R² = 1 - (Model Hatası / Toplam Varyans)\n",
    "\n",
    "Eğer varyans ≈ 0 → R² ≈ -sonsuz\n",
    "Senin doğrulama ve test veri setin neredeyse sabit, bu yüzden:\n",
    "\n",
    "R²: -3481 (validation)\n",
    "\n",
    "R²: -10145 (test)\n",
    "\n",
    "Bu aslında modelin kötü olduğunu değil, ölçüm yönteminin uygun olmadığını gösteriyor.\n",
    "\n",
    "✅ Ne Yapabilirsin?\n",
    "1. R² yerine MAE, MSE, RMSE kullan\n",
    "Bunlar sabit veriyle bile çalışır\n",
    "\n",
    "Daha güvenilirdir\n",
    "\n",
    "2. Doğrulama ve test setini yeniden seç\n",
    "Eğitim verisiyle benzer dağılıma sahip olmalı\n",
    "\n",
    "Örneğin: train_test_split(..., stratify=binned_targets) şeklinde\n",
    "\n",
    "3. Görsel inceleme yap\n",
    "y_true ve y_pred scatter plot veya line plot ile çiz\n",
    "\n",
    "Hataları görsel olarak fark edebilirsin\n",
    "\n",
    "4. Eğitim verisindeki dağılımı korumaya çalış\n",
    "Mümkünse tüm veriyi karıştırıp tekrar böl"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "adab9192-726a-4bb4-9d3e-51c50a4b69de",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(4963, 67) (709, 67) (631, 67) (4963,) (709,) (631,)\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# First, split into training + validation and testing sets\n",
    "X_train_val, X_test, y_train_val, y_test = train_test_split(X_battery, y_battery, test_size=0.1, random_state=42)\n",
    "\n",
    "# Then, split the training + validation set into training and validation sets\n",
    "X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42)  # 0.125 * 0.9 = 0.1 (10% for validation)\n",
    "\n",
    "# Check the split sizes\n",
    "print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "9b5d8ad6-be04-48ed-8a2a-7ccd48783e9e",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_tensor = torch.tensor(X_train, dtype=torch.float32)\n",
    "X_val_tensor = torch.tensor(X_val, dtype=torch.float32)\n",
    "X_test_tensor = torch.tensor(X_test, dtype=torch.float32)\n",
    "y_train_tensor = torch.tensor(y_train, dtype=torch.float32)\n",
    "y_val_tensor = torch.tensor(y_val, dtype=torch.float32)\n",
    "y_test_tensor = torch.tensor(y_test, dtype=torch.float32)\n",
    "\n",
    "# Move tensors to GPU if available\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "X_train_tensor = X_train_tensor.to(device)\n",
    "X_val_tensor = X_val_tensor.to(device)\n",
    "X_test_tensor = X_test_tensor.to(device)\n",
    "y_train_tensor = y_train_tensor.to(device)\n",
    "y_val_tensor = y_val_tensor.to(device)\n",
    "y_test_tensor = y_test_tensor.to(device)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "7d6cb763-9fb9-4e87-8088-d1419056d85f",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor), batch_size=64, shuffle=True)\n",
    "val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor), batch_size=64, shuffle=False)\n",
    "test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "fad8f9b1-ea3b-4a34-8a0d-475fc864cfaa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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lYulpaXJ3tSo6ijTre7Hjh2LHTt24OLFi9Dr9ahduzbWr1+Pfv36AQDOnz8Pf39/JCUloXXr1ti9ezd69OiBmzdvSkdVli9fjkmTJuH27duwsbEp0X55q3sikhve6p7o6SrkVve5ubn46quv8Oabb0KhUOD48eN48OABgoODpT5+fn6oU6cOkpKSAABJSUlo3LixwSmfkJAQ6PV6nDlz5rH7ysnJgV6vN3gQERFR1VXqgLJ161ZkZGRgyJAhAACdTgcbGxvpY6MLubm5QafTSX0eDieFywuXPU5sbCzUarX08PT0LG3ZREREVAmUOqCsWrUK3bp1g4eHhynrKVZ0dDQyMzOlx/Xr18t9n0RERGQ+pbp05tq1a9i3bx++++47qU2j0SA3NxcZGRkGR1HS0tKky400Gg2OHj1qsK20tDRp2eMolUoolcrSlEpERESVUKmOoMTHx8PV1RWhoaFSW2BgIKytrZGYmCi1JScnIzU1FVqtFgCg1Wpx6tQpg2veExISoFKpEBAQUNoxEBERURVj9BGUgoICxMfHIyIiwuDeJWq1GkOHDsX48ePh4uIClUqFMWPGQKvVonXr1gCALl26ICAgAIMHD0ZcXBx0Oh0mT56MyMhIHiEhIiIiidEBZd++fUhNTcWbb75ZZNm8efNgYWGBsLAwgxu1FbK0tMSOHTswatQoaLVaODg4ICIiAjExMWUbBREREVUpZboPirnwPihEJDe8DwrR01XIfVCIiIiIygsDChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyQ4DChEREckOAwoRERHJDgMKERERyY7RAeXPP//Ea6+9hpo1a8LOzg6NGzfGr7/+Ki0XQmDKlClwd3eHnZ0dgoODcfHiRYNt3L17F+Hh4VCpVHB2dsbQoUORlZVV9tEQERFRlWBUQPnf//6Htm3bwtraGrt378bZs2fx6aefokaNGlKfuLg4LFy4EMuXL8eRI0fg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oBAREZHsMKAQERGR7DCgEBERkewwoBAREZHsMKAQERGR7DCgEBERkewwoBAREZHsMKAQERGR7DCgEBERkewwoBAREZHsMKAQERGR7FiZuwA5mpdwodTrjuvcwISVEBERVU88gkJERESyw4BCREREssOAQkRERLLDgEJERESyw4BCREREssOreIioSinLVXhEJB88gkJERESyw4BCREREssOAQkRERLLDgEJERESyw4BCREREssOAQkRERLJjVEBZtmwZmjRpApVKBZVKBa1Wi927d0vLs7OzERkZiZo1a8LR0RFhYWFIS0sz2EZqaipCQ0Nhb28PV1dXTJw4EXl5eaYZDREREVUJRgWUZ599FrNmzcLx48fx66+/omPHjujduzfOnDkDABg3bhy2b9+OTZs24ccff8TNmzfRt29faf38/HyEhoYiNzcXhw4dwpo1a7B69WpMmTLFtKMiIiKiSk0hhBBl2YCLiwtmz56Nfv36oXbt2li/fj369esHADh//jz8/f2RlJSE1q1bY/fu3ejRowdu3rwJNzc3AMDy5csxadIk3L59GzY2NiXap16vh1qtRmZmJlQqVVnKL1ZZbvQ0rnMDE1ZCRMaqjDdq498Nqi6Mef8u9RyU/Px8bNiwAffv34dWq8Xx48fx4MEDBAcHS338/PxQp04dJCUlAQCSkpLQuHFjKZwAQEhICPR6vXQUpjg5OTnQ6/UGDyIiIqq6jA4op06dgqOjI5RKJf7zn/9gy5YtCAgIgE6ng42NDZydnQ36u7m5QafTAQB0Op1BOClcXrjscWJjY6FWq6WHp6ensWUTERFRJWJ0QHnuuedw8uRJHDlyBKNGjUJERATOnj1bHrVJoqOjkZmZKT2uX79ervsjIiIi8zL6wwJtbGxQv359AEBgYCCOHTuGBQsWYMCAAcjNzUVGRobBUZS0tDRoNBoAgEajwdGjRw22V3iVT2Gf4iiVSiiVSmNLJSIz4lwuIiqLMt8HpaCgADk5OQgMDIS1tTUSExOlZcnJyUhNTYVWqwUAaLVanDp1Cunp6VKfhIQEqFQqBAQElLUUIiIiqiKMOoISHR2Nbt26oU6dOrh37x7Wr1+PH374AXv37oVarcbQoUMxfvx4uLi4QKVSYcyYMdBqtWjdujUAoEuXLggICMDgwYMRFxcHnU6HyZMnIzIykkdIiIiISGJUQElP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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "plt.hist(y_train, bins=30, alpha=0.5, label='Train')\n",
    "plt.hist(y_val, bins=30, alpha=0.5, label='Validation')\n",
    "plt.hist(y_test, bins=30, alpha=0.5, label='Test')\n",
    "plt.legend()\n",
    "plt.title(\"Target Distribution Comparison\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "1fa220bc-81b0-411c-b2d2-c8263c1c1140",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "torch.Size([4963, 67]) torch.Size([4963])\n"
     ]
    }
   ],
   "source": [
    "print(X_train_tensor.shape,y_train_tensor.shape)\n",
    "dataset = TensorDataset(X_train_tensor, y_train_tensor)\n",
    "loader = DataLoader(dataset, batch_size=64, shuffle=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "04d01535-4169-45fe-b420-eb4820848c28",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [1/50], Train Loss: 0.0227, Train MAE: 0.1192, Train MSE: 0.0227, Train RMSE: 0.1502, Train R²: 0.0525\n",
      "Validation - Loss: 0.0240, MAE: 0.1182, MSE: 0.0240, RMSE: 0.1534, R²: 0.0473\n",
      "Epoch [2/50], Train Loss: 0.0226, Train MAE: 0.1185, Train MSE: 0.0226, Train RMSE: 0.1497, Train R²: 0.0531\n",
      "Validation - Loss: 0.0238, MAE: 0.1216, MSE: 0.0238, RMSE: 0.1529, R²: 0.0536\n",
      "Epoch [3/50], Train Loss: 0.0228, Train MAE: 0.1190, Train MSE: 0.0228, Train RMSE: 0.1506, Train R²: 0.0474\n",
      "Validation - Loss: 0.0238, MAE: 0.1214, MSE: 0.0238, RMSE: 0.1528, R²: 0.0548\n",
      "Epoch [4/50], Train Loss: 0.0228, Train MAE: 0.1188, Train MSE: 0.0228, Train RMSE: 0.1503, Train R²: 0.0463\n",
      "Validation - Loss: 0.0238, MAE: 0.1215, MSE: 0.0238, RMSE: 0.1527, R²: 0.0551\n",
      "Epoch [5/50], Train Loss: 0.0227, Train MAE: 0.1186, Train MSE: 0.0227, Train RMSE: 0.1500, Train R²: 0.0423\n",
      "Validation - Loss: 0.0237, MAE: 0.1193, MSE: 0.0237, RMSE: 0.1525, R²: 0.0576\n",
      "Epoch [6/50], Train Loss: 0.0227, Train MAE: 0.1190, Train MSE: 0.0227, Train RMSE: 0.1498, Train R²: 0.0498\n",
      "Validation - Loss: 0.0238, MAE: 0.1206, MSE: 0.0238, RMSE: 0.1526, R²: 0.0568\n",
      "Epoch [7/50], Train Loss: 0.0224, Train MAE: 0.1181, Train MSE: 0.0224, Train RMSE: 0.1493, Train R²: 0.0527\n",
      "Validation - Loss: 0.0237, MAE: 0.1200, MSE: 0.0237, RMSE: 0.1525, R²: 0.0580\n",
      "Epoch [8/50], Train Loss: 0.0224, Train MAE: 0.1179, Train MSE: 0.0224, Train RMSE: 0.1490, Train R²: 0.0562\n",
      "Validation - Loss: 0.0241, MAE: 0.1170, MSE: 0.0241, RMSE: 0.1539, R²: 0.0406\n",
      "Epoch [9/50], Train Loss: 0.0227, Train MAE: 0.1185, Train MSE: 0.0227, Train RMSE: 0.1501, Train R²: 0.0497\n",
      "Validation - Loss: 0.0238, MAE: 0.1220, MSE: 0.0238, RMSE: 0.1527, R²: 0.0548\n",
      "Epoch [10/50], Train Loss: 0.0226, Train MAE: 0.1184, Train MSE: 0.0226, Train RMSE: 0.1495, Train R²: 0.0557\n",
      "Validation - Loss: 0.0237, MAE: 0.1201, MSE: 0.0237, RMSE: 0.1524, R²: 0.0592\n",
      "Epoch [11/50], Train Loss: 0.0224, Train MAE: 0.1183, Train MSE: 0.0224, Train RMSE: 0.1493, Train R²: 0.0608\n",
      "Validation - Loss: 0.0238, MAE: 0.1197, MSE: 0.0238, RMSE: 0.1527, R²: 0.0550\n",
      "Epoch [12/50], Train Loss: 0.0225, Train MAE: 0.1179, Train MSE: 0.0225, Train RMSE: 0.1495, Train R²: 0.0549\n",
      "Validation - Loss: 0.0239, MAE: 0.1236, MSE: 0.0239, RMSE: 0.1532, R²: 0.0489\n",
      "Epoch [13/50], Train Loss: 0.0226, Train MAE: 0.1186, Train MSE: 0.0226, Train RMSE: 0.1500, Train R²: 0.0533\n",
      "Validation - Loss: 0.0239, MAE: 0.1236, MSE: 0.0239, RMSE: 0.1532, R²: 0.0490\n",
      "Epoch [14/50], Train Loss: 0.0228, Train MAE: 0.1192, Train MSE: 0.0228, Train RMSE: 0.1504, Train R²: 0.0449\n",
      "Validation - Loss: 0.0242, MAE: 0.1256, MSE: 0.0242, RMSE: 0.1540, R²: 0.0386\n",
      "Epoch [15/50], Train Loss: 0.0225, Train MAE: 0.1183, Train MSE: 0.0225, Train RMSE: 0.1495, Train R²: 0.0530\n",
      "Validation - Loss: 0.0240, MAE: 0.1241, MSE: 0.0240, RMSE: 0.1533, R²: 0.0473\n",
      "Epoch [16/50], Train Loss: 0.0226, Train MAE: 0.1179, Train MSE: 0.0226, Train RMSE: 0.1496, Train R²: 0.0529\n",
      "Validation - Loss: 0.0237, MAE: 0.1207, MSE: 0.0237, RMSE: 0.1524, R²: 0.0592\n",
      "Epoch [17/50], Train Loss: 0.0226, Train MAE: 0.1179, Train MSE: 0.0226, Train RMSE: 0.1498, Train R²: 0.0534\n",
      "Validation - Loss: 0.0237, MAE: 0.1194, MSE: 0.0237, RMSE: 0.1525, R²: 0.0580\n",
      "Epoch [18/50], Train Loss: 0.0226, Train MAE: 0.1183, Train MSE: 0.0226, Train RMSE: 0.1497, Train R²: 0.0518\n",
      "Validation - Loss: 0.0238, MAE: 0.1219, MSE: 0.0238, RMSE: 0.1526, R²: 0.0562\n",
      "Epoch [19/50], Train Loss: 0.0225, Train MAE: 0.1177, Train MSE: 0.0225, Train RMSE: 0.1493, Train R²: 0.0595\n",
      "Validation - Loss: 0.0241, MAE: 0.1171, MSE: 0.0241, RMSE: 0.1537, R²: 0.0432\n",
      "Epoch [20/50], Train Loss: 0.0224, Train MAE: 0.1175, Train MSE: 0.0224, Train RMSE: 0.1489, Train R²: 0.0622\n",
      "Validation - Loss: 0.0237, MAE: 0.1202, MSE: 0.0237, RMSE: 0.1524, R²: 0.0589\n",
      "Epoch [21/50], Train Loss: 0.0224, Train MAE: 0.1178, Train MSE: 0.0224, Train RMSE: 0.1492, Train R²: 0.0553\n",
      "Validation - Loss: 0.0237, MAE: 0.1187, MSE: 0.0237, RMSE: 0.1523, R²: 0.0603\n",
      "Epoch [22/50], Train Loss: 0.0225, Train MAE: 0.1179, Train MSE: 0.0225, Train RMSE: 0.1496, Train R²: 0.0537\n",
      "Validation - Loss: 0.0238, MAE: 0.1217, MSE: 0.0238, RMSE: 0.1526, R²: 0.0559\n",
      "Epoch [23/50], Train Loss: 0.0223, Train MAE: 0.1178, Train MSE: 0.0223, Train RMSE: 0.1488, Train R²: 0.0626\n",
      "Validation - Loss: 0.0239, MAE: 0.1237, MSE: 0.0239, RMSE: 0.1532, R²: 0.0492\n",
      "Epoch [24/50], Train Loss: 0.0222, Train MAE: 0.1172, Train MSE: 0.0222, Train RMSE: 0.1484, Train R²: 0.0666\n",
      "Validation - Loss: 0.0238, MAE: 0.1192, MSE: 0.0238, RMSE: 0.1526, R²: 0.0561\n",
      "Epoch [25/50], Train Loss: 0.0225, Train MAE: 0.1174, Train MSE: 0.0225, Train RMSE: 0.1495, Train R²: 0.0600\n",
      "Validation - Loss: 0.0242, MAE: 0.1256, MSE: 0.0242, RMSE: 0.1539, R²: 0.0393\n",
      "Epoch [26/50], Train Loss: 0.0224, Train MAE: 0.1177, Train MSE: 0.0224, Train RMSE: 0.1490, Train R²: 0.0547\n",
      "Validation - Loss: 0.0239, MAE: 0.1178, MSE: 0.0239, RMSE: 0.1530, R²: 0.0520\n",
      "Epoch [27/50], Train Loss: 0.0224, Train MAE: 0.1176, Train MSE: 0.0224, Train RMSE: 0.1491, Train R²: 0.0569\n",
      "Validation - Loss: 0.0248, MAE: 0.1296, MSE: 0.0248, RMSE: 0.1560, R²: 0.0118\n",
      "Epoch [28/50], Train Loss: 0.0226, Train MAE: 0.1179, Train MSE: 0.0226, Train RMSE: 0.1497, Train R²: 0.0526\n",
      "Validation - Loss: 0.0237, MAE: 0.1195, MSE: 0.0237, RMSE: 0.1525, R²: 0.0574\n",
      "Epoch [29/50], Train Loss: 0.0223, Train MAE: 0.1169, Train MSE: 0.0223, Train RMSE: 0.1487, Train R²: 0.0581\n",
      "Validation - Loss: 0.0236, MAE: 0.1196, MSE: 0.0236, RMSE: 0.1521, R²: 0.0632\n",
      "Epoch [30/50], Train Loss: 0.0226, Train MAE: 0.1181, Train MSE: 0.0226, Train RMSE: 0.1495, Train R²: 0.0491\n",
      "Validation - Loss: 0.0241, MAE: 0.1170, MSE: 0.0241, RMSE: 0.1536, R²: 0.0433\n",
      "Epoch [31/50], Train Loss: 0.0225, Train MAE: 0.1179, Train MSE: 0.0225, Train RMSE: 0.1494, Train R²: 0.0546\n",
      "Validation - Loss: 0.0236, MAE: 0.1193, MSE: 0.0236, RMSE: 0.1520, R²: 0.0645\n",
      "Epoch [32/50], Train Loss: 0.0224, Train MAE: 0.1175, Train MSE: 0.0224, Train RMSE: 0.1492, Train R²: 0.0585\n",
      "Validation - Loss: 0.0236, MAE: 0.1189, MSE: 0.0236, RMSE: 0.1520, R²: 0.0644\n",
      "Epoch [33/50], Train Loss: 0.0227, Train MAE: 0.1187, Train MSE: 0.0227, Train RMSE: 0.1501, Train R²: 0.0536\n",
      "Validation - Loss: 0.0236, MAE: 0.1182, MSE: 0.0236, RMSE: 0.1520, R²: 0.0648\n",
      "Epoch [34/50], Train Loss: 0.0226, Train MAE: 0.1181, Train MSE: 0.0226, Train RMSE: 0.1497, Train R²: 0.0540\n",
      "Validation - Loss: 0.0237, MAE: 0.1221, MSE: 0.0237, RMSE: 0.1524, R²: 0.0589\n",
      "Epoch [35/50], Train Loss: 0.0224, Train MAE: 0.1172, Train MSE: 0.0224, Train RMSE: 0.1489, Train R²: 0.0565\n",
      "Validation - Loss: 0.0235, MAE: 0.1190, MSE: 0.0235, RMSE: 0.1518, R²: 0.0665\n",
      "Epoch [36/50], Train Loss: 0.0226, Train MAE: 0.1181, Train MSE: 0.0226, Train RMSE: 0.1495, Train R²: 0.0519\n",
      "Validation - Loss: 0.0240, MAE: 0.1249, MSE: 0.0240, RMSE: 0.1534, R²: 0.0455\n",
      "Epoch [37/50], Train Loss: 0.0221, Train MAE: 0.1166, Train MSE: 0.0221, Train RMSE: 0.1482, Train R²: 0.0667\n",
      "Validation - Loss: 0.0236, MAE: 0.1179, MSE: 0.0236, RMSE: 0.1523, R²: 0.0608\n",
      "Epoch [38/50], Train Loss: 0.0224, Train MAE: 0.1175, Train MSE: 0.0224, Train RMSE: 0.1490, Train R²: 0.0649\n",
      "Validation - Loss: 0.0236, MAE: 0.1201, MSE: 0.0236, RMSE: 0.1521, R²: 0.0626\n",
      "Epoch [39/50], Train Loss: 0.0225, Train MAE: 0.1177, Train MSE: 0.0225, Train RMSE: 0.1494, Train R²: 0.0559\n",
      "Validation - Loss: 0.0236, MAE: 0.1197, MSE: 0.0236, RMSE: 0.1520, R²: 0.0637\n",
      "Epoch [40/50], Train Loss: 0.0222, Train MAE: 0.1172, Train MSE: 0.0222, Train RMSE: 0.1484, Train R²: 0.0689\n",
      "Validation - Loss: 0.0241, MAE: 0.1257, MSE: 0.0241, RMSE: 0.1538, R²: 0.0406\n",
      "Epoch [41/50], Train Loss: 0.0223, Train MAE: 0.1173, Train MSE: 0.0223, Train RMSE: 0.1488, Train R²: 0.0622\n",
      "Validation - Loss: 0.0238, MAE: 0.1227, MSE: 0.0238, RMSE: 0.1525, R²: 0.0570\n",
      "Epoch [42/50], Train Loss: 0.0221, Train MAE: 0.1169, Train MSE: 0.0221, Train RMSE: 0.1479, Train R²: 0.0652\n",
      "Validation - Loss: 0.0238, MAE: 0.1231, MSE: 0.0238, RMSE: 0.1527, R²: 0.0551\n",
      "Epoch [43/50], Train Loss: 0.0225, Train MAE: 0.1180, Train MSE: 0.0225, Train RMSE: 0.1493, Train R²: 0.0594\n",
      "Validation - Loss: 0.0237, MAE: 0.1218, MSE: 0.0237, RMSE: 0.1523, R²: 0.0602\n",
      "Epoch [44/50], Train Loss: 0.0221, Train MAE: 0.1174, Train MSE: 0.0221, Train RMSE: 0.1483, Train R²: 0.0707\n",
      "Validation - Loss: 0.0243, MAE: 0.1271, MSE: 0.0243, RMSE: 0.1545, R²: 0.0320\n",
      "Epoch [45/50], Train Loss: 0.0224, Train MAE: 0.1176, Train MSE: 0.0224, Train RMSE: 0.1491, Train R²: 0.0571\n",
      "Validation - Loss: 0.0237, MAE: 0.1185, MSE: 0.0237, RMSE: 0.1523, R²: 0.0603\n",
      "Epoch [46/50], Train Loss: 0.0222, Train MAE: 0.1172, Train MSE: 0.0222, Train RMSE: 0.1483, Train R²: 0.0696\n",
      "Validation - Loss: 0.0236, MAE: 0.1193, MSE: 0.0236, RMSE: 0.1519, R²: 0.0652\n",
      "Epoch [47/50], Train Loss: 0.0223, Train MAE: 0.1171, Train MSE: 0.0223, Train RMSE: 0.1484, Train R²: 0.0663\n",
      "Validation - Loss: 0.0237, MAE: 0.1224, MSE: 0.0237, RMSE: 0.1523, R²: 0.0596\n",
      "Epoch [48/50], Train Loss: 0.0223, Train MAE: 0.1176, Train MSE: 0.0223, Train RMSE: 0.1488, Train R²: 0.0629\n",
      "Validation - Loss: 0.0238, MAE: 0.1232, MSE: 0.0238, RMSE: 0.1526, R²: 0.0563\n",
      "Epoch [49/50], Train Loss: 0.0223, Train MAE: 0.1174, Train MSE: 0.0223, Train RMSE: 0.1487, Train R²: 0.0628\n",
      "Validation - Loss: 0.0237, MAE: 0.1224, MSE: 0.0237, RMSE: 0.1523, R²: 0.0596\n",
      "Epoch [50/50], Train Loss: 0.0226, Train MAE: 0.1180, Train MSE: 0.0226, Train RMSE: 0.1496, Train R²: 0.0611\n",
      "Validation - Loss: 0.0235, MAE: 0.1196, MSE: 0.0235, RMSE: 0.1519, R²: 0.0655\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# Assuming your fine_tuned_model, criterion (e.g., MSELoss), and optimizer are already defined\n",
    "num_epochs = 50\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "fine_tuned_model.to(device)\n",
    "\n",
    "# Function to calculate R-squared (R²)\n",
    "def r2_score(y_true, y_pred):\n",
    "    ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)\n",
    "    ss_residual = torch.sum((y_true - y_pred) ** 2)\n",
    "    return 1 - (ss_residual / ss_total)\n",
    "\n",
    "# Function to evaluate the model\n",
    "def evaluate_model(model, loader, device):\n",
    "    model.eval()  # Set model to evaluation mode\n",
    "    epoch_loss = 0.0\n",
    "    epoch_mae = 0.0\n",
    "    epoch_mse = 0.0\n",
    "    epoch_rmse = 0.0\n",
    "    epoch_r2 = 0.0\n",
    "\n",
    "    with torch.no_grad():  # Disable gradient computation for evaluation\n",
    "        for X_batch, y_batch in loader:\n",
    "            X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n",
    "            outputs = model(X_batch)\n",
    "\n",
    "            # Ensure output and target shapes match\n",
    "            if outputs.shape != y_batch.shape:\n",
    "                y_batch = y_batch.view_as(outputs)\n",
    "\n",
    "            # Calculate metrics\n",
    "            loss = criterion(outputs, y_batch)\n",
    "            mae = torch.mean(torch.abs(outputs - y_batch))  # Mean Absolute Error\n",
    "            mse = F.mse_loss(outputs, y_batch)  # Mean Squared Error\n",
    "            rmse = torch.sqrt(mse)  # Root Mean Squared Error\n",
    "            r2 = r2_score(y_batch, outputs)  # R-squared\n",
    "\n",
    "            # Accumulate metrics for the epoch\n",
    "            epoch_loss += loss.item()\n",
    "            epoch_mae += mae.item()\n",
    "            epoch_mse += mse.item()\n",
    "            epoch_rmse += rmse.item()\n",
    "            epoch_r2 += r2.item()\n",
    "\n",
    "    # Return average metrics for the epoch\n",
    "    return epoch_loss / len(loader), epoch_mae / len(loader), epoch_mse / len(loader), epoch_rmse / len(loader), epoch_r2 / len(loader)\n",
    "\n",
    "# Training loop\n",
    "for epoch in range(num_epochs):\n",
    "    fine_tuned_model.train()  # Set model to training mode\n",
    "    epoch_loss = 0.0\n",
    "    epoch_mae = 0.0\n",
    "    epoch_mse = 0.0\n",
    "    epoch_rmse = 0.0\n",
    "    epoch_r2 = 0.0\n",
    "    \n",
    "    # Training phase\n",
    "    for X_batch, y_batch in loader:\n",
    "        X_batch, y_batch = X_batch.to(device), y_batch.to(device)\n",
    "\n",
    "        optimizer.zero_grad()  # Zero the gradients\n",
    "        outputs = fine_tuned_model(X_batch)  # Forward pass\n",
    "\n",
    "        # Ensure output and target shapes match\n",
    "        if outputs.shape != y_batch.shape:\n",
    "            y_batch = y_batch.view_as(outputs)\n",
    "\n",
    "        # Calculate loss\n",
    "        loss = criterion(outputs, y_batch)  # Calculate MSE loss (since criterion is usually MSELoss)\n",
    "\n",
    "        # Calculate MAE and MSE\n",
    "        mae = torch.mean(torch.abs(outputs - y_batch))  # Mean Absolute Error\n",
    "        mse = F.mse_loss(outputs, y_batch)  # Mean Squared Error\n",
    "        rmse = torch.sqrt(mse)  # Root Mean Squared Error (RMSE)\n",
    "        r2 = r2_score(y_batch, outputs)  # R-squared (R²)\n",
    "\n",
    "        loss.backward()  # Backward pass\n",
    "        optimizer.step()  # Update weights\n",
    "\n",
    "        # Accumulate loss and metrics for the epoch\n",
    "        epoch_loss += loss.item()\n",
    "        epoch_mae += mae.item()\n",
    "        epoch_mse += mse.item()\n",
    "        epoch_rmse += rmse.item()\n",
    "        epoch_r2 += r2.item()\n",
    "\n",
    "    # Print the average training metrics for the epoch\n",
    "    print(f\"Epoch [{epoch+1}/{num_epochs}], \"\n",
    "          f\"Train Loss: {epoch_loss/len(loader):.4f}, \"\n",
    "          f\"Train MAE: {epoch_mae/len(loader):.4f}, \"\n",
    "          f\"Train MSE: {epoch_mse/len(loader):.4f}, \"\n",
    "          f\"Train RMSE: {epoch_rmse/len(loader):.4f}, \"\n",
    "          f\"Train R²: {epoch_r2/len(loader):.4f}\")\n",
    "\n",
    "    # Validation phase after each epoch\n",
    "    val_loss, val_mae, val_mse, val_rmse, val_r2 = evaluate_model(fine_tuned_model, val_loader, device)\n",
    "    print(f\"Validation - Loss: {val_loss:.4f}, \"\n",
    "          f\"MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, \"\n",
    "          f\"RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "20539336-3d9d-411d-b1aa-e5e0fe322feb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test - Loss: 0.0224, MAE: 0.1168, MSE: 0.0224, RMSE: 0.1491, R²: 0.0879\n"
     ]
    }
   ],
   "source": [
    "test_loss, test_mae, test_mse, test_rmse, test_r2 = evaluate_model(fine_tuned_model, test_loader, device)\n",
    "print(f\"Test - Loss: {test_loss:.4f}, \"\n",
    "      f\"MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, \"\n",
    "      f\"RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "904aa414-ea92-4c41-98d1-6ba58dfe781a",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "69a3312d-a53b-4db0-a3a8-5d7b3e4ddf70",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
