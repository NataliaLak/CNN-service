import os
import torch
import torch.nn as nn

# Определяем модель
class CNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=n_classes)
        )

    def forward(self, x):
        return self.model(x)
    
# Определяем модель
class Model:
    def __init__(self):
        # Загружаем модель
        self.model = CNN(47)
        model_path = os.path.join('myapp', 'model.ckpt')
        self.model.load_state_dict(torch.load(model_path))  
        self.model.eval()  # Переводим модель в режим оценки

        # Загружаем соответствия между метками и символами
        self.label_to_char = self.load_mapping('data/EMNIST/raw/emnist-balanced-mapping.txt')

    def load_mapping(self, mapping_path):
        label_to_char = {}
        with open(mapping_path, 'r') as f:
            for line in f:
                label, char = line.strip().split()
                label_to_char[int(label)] = chr(int(char))
        return label_to_char
    
    def predict(self, x):
        '''
        Parameters
        ----------
        x : torch.Tensor
            Входное изображение -- тензор размера (1, 28, 28)
        Returns
        -------
        pred : str
            Символ-предсказание 
        '''
        x = x.float()  
        if x.dim() == 2:  
            x = x.unsqueeze(0)  
            x = x.unsqueeze(0)  
        
        with torch.no_grad():
            output = self.model(x)
            pred_label = output.argmax(dim=1).item()  # Получаем предсказанный лейбл

        return self.label_to_char[pred_label]  # Возвращаем соответствующий символ